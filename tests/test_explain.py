# Copyright: See the LICENSE file.

"""Tests for the static declaration override conflict report."""

import json
import unittest

import factory
from factory import declarations
from factory.explain import (
    CycleRiskKind,
    OverrideReport,
    OverrideSource,
    explain_overrides,
)


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Module-level, mutually recursive factories (string paths resolve through
# the normal import machinery, mirroring the tests/cyclic helpers).
class _NodeFactory(factory.Factory):
    class Meta:
        model = Dummy
    parent = factory.SubFactory('%s._NodeFactory' % __name__)


class _XFactory(factory.Factory):
    class Meta:
        model = Dummy
    y = factory.SubFactory('%s._YFactory' % __name__)


class _YFactory(factory.Factory):
    class Meta:
        model = Dummy
    x = factory.SubFactory('%s._XFactory' % __name__)


class ExplainerAPITestCase(unittest.TestCase):
    def test_module_function_and_classmethod_equivalent(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1

        report = explain_overrides(MyFactory)
        self.assertIsInstance(report, OverrideReport)
        self.assertIsInstance(MyFactory.explain_overrides(), OverrideReport)
        self.assertEqual(
            report.as_dict(), MyFactory.explain_overrides().as_dict())

    def test_requires_factory_subclass(self):
        with self.assertRaises(TypeError):
            explain_overrides(Dummy)

    def test_sorted_field_names_and_iteration(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            zeta = 1
            alpha = 2
            middle = 3

        report = explain_overrides(MyFactory)
        self.assertEqual(
            report.sorted_field_names(), ['alpha', 'middle', 'zeta'])
        self.assertEqual(list(report), ['alpha', 'middle', 'zeta'])

    def test_as_dict_is_json_friendly(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1

        report = explain_overrides(MyFactory, one=2)
        rendered = json.dumps(report.as_dict())
        parsed = json.loads(rendered)
        self.assertEqual(list(parsed['fields']), ['one'])


class PrecedenceChainTestCase(unittest.TestCase):
    def test_base_param_trait_kwarg_chain(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            a = 1

            class Params:
                flag = factory.Trait(a=2)

        report = explain_overrides(MyFactory, a=9)
        field = report['a']
        self.assertEqual(
            [step.source for step in field.chain],
            [OverrideSource.BASE, OverrideSource.TRAIT,
             OverrideSource.KWARG],
        )
        self.assertEqual(field.winner.source, OverrideSource.KWARG)
        self.assertEqual(field.winner.kind, 'int')
        # The base value is only dropped on the active trait branch.
        self.assertEqual(field.abandoned[0].conditional_on, 'flag')
        self.assertEqual(field.abandoned[0].reason, 'replaced')
        self.assertIsNone(field.abandoned[1].conditional_on)
        self.assertIn('a', report.conflicts)

    def test_simple_param_wins_over_base(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            zero = 0
            value = factory.LazyAttribute(lambda o: o.zero + 1)

            class Params:
                zero = 1

        report = explain_overrides(MyFactory)
        zero = report['zero']
        self.assertEqual(
            [step.source for step in zero.chain],
            [OverrideSource.BASE, OverrideSource.PARAM])
        # Simple Params are stripped before reaching the model.
        self.assertTrue(zero.parameter_flag)
        self.assertFalse(zero.delivered)
        self.assertEqual(
            [a.reason for a in zero.abandoned], ['replaced', 'hidden_param'])
        # Kinds expose structural types, never the values.
        self.assertEqual(zero.winner.kind, 'int')
        self.assertEqual(report['value'].winner.kind, 'LazyAttribute')

    def test_inherited_declaration_provenance(self):
        class ParentFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1

        class ChildFactory(ParentFactory):
            two = 2

        report = explain_overrides(ChildFactory)
        self.assertEqual(report['one'].winner.source,
                         OverrideSource.INHERITED)
        self.assertEqual(report['one'].winner.origin, 'ParentFactory')
        self.assertEqual(report['two'].winner.source,
                         OverrideSource.BASE)

        class OverridingFactory(ParentFactory):
            one = 3

        report2 = explain_overrides(OverridingFactory)
        chain = report2['one'].chain
        self.assertEqual(
            [(s.source, s.origin) for s in chain],
            [(OverrideSource.INHERITED, 'ParentFactory'),
             (OverrideSource.BASE, 'OverridingFactory')])

    def test_runtime_parity_for_unconditional_layers(self):
        sentinel_base = object()
        sentinel_kwarg = object()
        sentinel_default = object()
        sentinel_context = object()

        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = sentinel_base

        class PlainOuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            inner = factory.SubFactory(InnerFactory)

        class DefaultOuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            inner = factory.SubFactory(
                InnerFactory, c=sentinel_default)

        # Nested factory declaration alone.
        obj = PlainOuterFactory.build()
        self.assertIs(obj.inner.c, sentinel_base)
        report = explain_overrides(PlainOuterFactory)
        self.assertEqual(
            report['inner'].subfactory['c'].winner.source,
            OverrideSource.BASE)

        # Sub-factory default beats nested factory declaration.
        obj = DefaultOuterFactory.build()
        self.assertIs(obj.inner.c, sentinel_default)
        report = explain_overrides(DefaultOuterFactory)
        nested = report['inner'].subfactory
        self.assertEqual(nested['c'].winner.source,
                         OverrideSource.SUBFACTORY_DEFAULT)

        # Outer call-time context beats the sub-factory default.
        obj = DefaultOuterFactory.build(inner__c=sentinel_context)
        self.assertIs(obj.inner.c, sentinel_context)
        report = explain_overrides(
            DefaultOuterFactory, inner__c=sentinel_context)
        nested = report['inner'].subfactory
        self.assertEqual(nested['c'].winner.source,
                         OverrideSource.SUBFACTORY_CONTEXT)

        # Flat kwarg on the outer field replaces the whole sub-factory.
        obj = DefaultOuterFactory.build(inner=sentinel_kwarg)
        self.assertIs(obj.inner, sentinel_kwarg)
        report = explain_overrides(
            DefaultOuterFactory, inner=sentinel_kwarg)
        self.assertEqual(report['inner'].winner.source,
                         OverrideSource.KWARG)
        self.assertIsNone(report['inner'].subfactory)


class SameNameTraitTestCase(unittest.TestCase):
    def test_multiple_traits_target_same_field(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            y = 0

            class Params:
                t1 = factory.Trait(y=11)
                t2 = factory.Trait(y=12)

        report = explain_overrides(MyFactory)
        chain = report['y'].chain
        self.assertEqual(
            [s.source for s in chain],
            [OverrideSource.BASE, OverrideSource.TRAIT,
             OverrideSource.TRAIT])
        self.assertEqual(
            [s.conditional_on for s in chain[1:]], ['t1', 't2'])
        self.assertEqual(chain[-1].origin, 'MyFactory.Params.t2')

    def test_trait_shares_name_with_declared_field(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            # Falsy default keeps the trait inactive; a truthy declaration
            # here would enable the trait by default at runtime.
            x = 0
            y = 0

            class Params:
                # The flag is named like a real field; it only drives the
                # Maybe and is stripped before instantiation.
                x = factory.Trait(y=10)

        report = explain_overrides(MyFactory)
        x_field = report['x']
        self.assertTrue(x_field.parameter_flag)
        self.assertFalse(x_field.delivered)
        self.assertEqual(
            [a.reason for a in x_field.abandoned], ['hidden_param'])
        y_field = report['y']
        self.assertEqual(y_field.chain[1].conditional_on, 'x')

        # And runtime agrees: flag drives y, x itself is not passed.
        obj = MyFactory.build()
        self.assertFalse(hasattr(obj, 'x'))
        self.assertEqual(obj.y, 0)
        obj = MyFactory.build(x=True)
        self.assertEqual(obj.y, 10)

    def test_subclass_redefines_same_name_trait(self):
        class ParentFactory(factory.Factory):
            class Meta:
                model = Dummy
            y = 0

            class Params:
                flag = factory.Trait(y=1)

        class ChildFactory(ParentFactory):
            class Params:
                flag = factory.Trait(y=2)

        report = explain_overrides(ChildFactory)
        winner = report['y'].winner
        self.assertEqual(winner.source, OverrideSource.TRAIT)
        self.assertEqual(winner.origin, 'ChildFactory.Params.flag')

    def test_empty_trait_is_recorded(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1

            class Params:
                nothing = factory.Trait()

        report = explain_overrides(MyFactory)
        self.assertIn('MyFactory.Params.nothing', report.empty_traits)
        self.assertTrue(report['nothing'].parameter_flag)

    def test_trait_flag_only_field(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy

            class Params:
                flag = factory.Trait(other=1)
            other = 0

        report = explain_overrides(MyFactory)
        flag = report['flag']
        self.assertEqual(flag.chain, [])
        self.assertIsNone(flag.winner)
        self.assertTrue(flag.parameter_flag)
        self.assertFalse(flag.delivered)


class SubFactoryTestCase(unittest.TestCase):
    def test_double_layer_context(self):
        class CFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = 3

        class BFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = factory.SubFactory(CFactory)

        class AFactory(factory.Factory):
            class Meta:
                model = Dummy
            b = factory.SubFactory(BFactory)

        report = explain_overrides(AFactory, **{'b__c__c': 7})
        b_report = report['b'].subfactory
        self.assertEqual(b_report.factory, BFactory)
        c_report = b_report['c'].subfactory
        self.assertEqual(c_report.factory, CFactory)
        chain = c_report['c'].chain
        self.assertEqual(
            [s.source for s in chain],
            [OverrideSource.BASE, OverrideSource.SUBFACTORY_CONTEXT])

    def test_double_layer_defaults(self):
        class CFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = 3

        class BFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = factory.SubFactory(CFactory)

        class AFactory(factory.Factory):
            class Meta:
                model = Dummy
            b = factory.SubFactory(
                BFactory, c=factory.SubFactory(CFactory, c=5))

        report = explain_overrides(AFactory)
        b_report = report['b'].subfactory
        self.assertEqual(b_report['c'].winner.source,
                         OverrideSource.SUBFACTORY_DEFAULT)
        c_report = b_report['c'].subfactory
        self.assertEqual(c_report['c'].winner.source,
                         OverrideSource.SUBFACTORY_DEFAULT)
        self.assertEqual(c_report['c'].chain[0].source,
                         OverrideSource.BASE)

    def test_subfactory_without_overrides(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = 3

        class OuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            inner = factory.SubFactory(InnerFactory)

        report = explain_overrides(OuterFactory)
        nested = report['inner'].subfactory
        self.assertEqual(nested.factory, InnerFactory)
        self.assertEqual(
            nested['c'].winner.source, OverrideSource.BASE)
        self.assertEqual(nested['c'].abandoned, [])

    def test_deep_trait_into_subfactory(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 0

        class OuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 0
            inner = factory.SubFactory(InnerFactory)

            class Params:
                deep_one = factory.Trait(one=1, inner__one=2)

        report = explain_overrides(OuterFactory)
        nested = report['inner'].subfactory
        winner = nested['one'].winner
        self.assertEqual(winner.source, OverrideSource.TRAIT)
        self.assertEqual(winner.conditional_on, 'deep_one')

    def test_trait_providing_subfactory_is_followed(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = 0
            y = 0

        class OuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            child = None

            class Params:
                with_child = factory.Trait(
                    child=factory.SubFactory(InnerFactory, x=1))

        report = explain_overrides(OuterFactory)
        nested = report['child'].subfactory
        self.assertIsNotNone(nested)
        self.assertEqual(nested.factory, InnerFactory)
        self.assertEqual(nested['x'].winner.source,
                         OverrideSource.SUBFACTORY_DEFAULT)
        # Runtime agrees.
        obj = OuterFactory.build(with_child=True)
        self.assertEqual(obj.child.x, 1)
        self.assertEqual(obj.child.y, 0)

    def test_subfactory_default_self_attribute_scoped_to_inner_factory(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = 0
            y = 0

        class OuterFactory(factory.Factory):
            class Meta:
                model = Dummy
            child = factory.SubFactory(
                InnerFactory, x=factory.SelfAttribute('y'))

        report = explain_overrides(OuterFactory)
        # The reference must not create a spurious outer-level cycle.
        self.assertEqual(report.cycle_risks, [])
        nested = report['child'].subfactory
        self.assertEqual(nested['x'].winner.kind, 'SelfAttribute')
        self.assertEqual(nested['x'].cycle_risks, [])
        # Runtime agrees: inner x reads inner y provided by the outer context.
        obj = OuterFactory.build(child__y=5)
        self.assertEqual(obj.child.x, 5)

    def test_self_referential_subfactory_cycle(self):
        report = explain_overrides(_NodeFactory)
        risks = report['parent'].cycle_risks
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].kind,
                         CycleRiskKind.SUBFACTORY_RECURSION)
        self.assertIn('_NodeFactory', risks[0].path[0])

        # Supplying the cyclic field at call time removes the risk.
        report = explain_overrides(_NodeFactory, parent=None)
        self.assertEqual(report['parent'].cycle_risks, [])
        self.assertIsNone(report['parent'].subfactory)

    def test_mutual_subfactory_cycle(self):
        report = explain_overrides(_XFactory)
        nested = report['y'].subfactory
        risks = nested['x'].cycle_risks
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].kind,
                         CycleRiskKind.SUBFACTORY_RECURSION)
        self.assertEqual(risks[0].path[:2], ['_XFactory', '_YFactory'])

    def test_dict_and_list_nesting(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1

        class HolderFactory(factory.Factory):
            class Meta:
                model = Dummy
            mapping = factory.Dict({'one': 1, 'two': factory.SubFactory(
                InnerFactory)})
            sequence = factory.List([0, factory.SubFactory(InnerFactory)])

        report = explain_overrides(HolderFactory)
        mapping = report['mapping'].subfactory
        self.assertEqual(mapping.sorted_field_names(), ['one', 'two'])
        self.assertIsNotNone(mapping['two'].subfactory)
        listed = report['sequence'].subfactory
        self.assertEqual(listed.sorted_field_names(), ['0', '1'])
        self.assertIsNotNone(listed['1'].subfactory)


class MetaOptionsTestCase(unittest.TestCase):
    def test_exclude(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
                exclude = ('secret',)
            secret = 'shh'
            public = 1

        report = explain_overrides(MyFactory, secret='override')
        secret = report['secret']
        self.assertTrue(secret.excluded)
        self.assertFalse(secret.delivered)
        self.assertIn('excluded', [a.reason for a in secret.abandoned])
        self.assertTrue(report['public'].delivered)

    def test_rename_shadows_target(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
                rename = {'attributes_': 'attributes'}
            attributes_ = 42
            attributes = 'old'

        report = explain_overrides(MyFactory)
        self.assertEqual(report['attributes_'].renamed_to, 'attributes')
        self.assertEqual(report['attributes'].renamed_from, 'attributes_')
        self.assertEqual(
            report['attributes'].shadowed_by_rename, 'attributes_')
        self.assertIn(
            'shadowed_by_rename',
            [a.reason for a in report['attributes'].abandoned])

    def test_rename_missing_source_is_not_an_error(self):
        # Mirrors https://github.com/FactoryBoy/factory_boy/issues/504
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
                rename = {'form_attributes': 'attributes'}

        report = explain_overrides(MyFactory)
        self.assertEqual(report.errors, [])
        self.assertIsNone(report['form_attributes'].winner)

    def test_inline_args(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
                inline_args = ('x',)
            x = 1
            y = 2

        report = explain_overrides(MyFactory)
        self.assertEqual(report['x'].inline_arg, 0)
        self.assertTrue(report['x'].delivered)
        self.assertIsNone(report['y'].inline_arg)

    def test_skip_declaration(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            removed = declarations.SKIP

        report = explain_overrides(MyFactory)
        field = report['removed']
        self.assertEqual(field.winner.kind, 'Skip')
        self.assertFalse(field.delivered)
        self.assertIn('skip', [a.reason for a in field.abandoned])


class TransformerAndPostGenerationTestCase(unittest.TestCase):
    def test_transformer_captures_flat_override(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = factory.Transformer(0, transform=lambda v: v + 1)

        report = explain_overrides(MyFactory, x=10)
        field = report['x']
        self.assertEqual(field.winner.kind, 'Transformer')
        self.assertEqual(
            [(p, s.source) for p, s in field.context_chain],
            [('', OverrideSource.KWARG)])

    def test_post_generation_value_goes_to_context(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy

            @factory.post_generation
            def hook(self, create, value, **kwargs):
                self.got = (value, kwargs)

        report = explain_overrides(MyFactory, hook=42, hook__extra=1)
        field = report['hook']
        self.assertEqual(field.winner.kind, 'PostGeneration')
        self.assertEqual(field.phase, 'post_instance')
        self.assertFalse(field.delivered)
        self.assertEqual(
            sorted(p for p, _ in field.context_chain), ['', 'extra'])

    def test_deep_override_of_unknown_field_is_reported_not_raised(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            a = 1

        report = explain_overrides(MyFactory, **{'b__c': 1})
        self.assertEqual(len(report.errors), 1)
        self.assertIn('b__c', report.errors[0])
        self.assertNotIn('b', report)


class EmptyOverrideTestCase(unittest.TestCase):
    def test_no_kwargs_report(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            one = 1
            two = factory.Sequence(lambda n: n)

        report = explain_overrides(MyFactory)
        self.assertEqual(report['one'].chain[0].source,
                         OverrideSource.BASE)
        self.assertEqual(report['one'].abandoned, [])
        self.assertEqual(report.conflicts, [])
        self.assertEqual(report.errors, [])

    def test_empty_trait_and_empty_subfactory_defaults(self):
        class InnerFactory(factory.Factory):
            class Meta:
                model = Dummy
            c = 1

        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            inner = factory.SubFactory(InnerFactory)

            class Params:
                unused = factory.Trait()

        report = explain_overrides(MyFactory)
        self.assertIn('MyFactory.Params.unused', report.empty_traits)
        nested = report['inner'].subfactory
        self.assertEqual(nested['c'].winner.source,
                         OverrideSource.BASE)


class CycleDetectionTestCase(unittest.TestCase):
    def test_self_attribute_cycle(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = factory.SelfAttribute('y')
            y = factory.SelfAttribute('x')

        report = explain_overrides(MyFactory)
        self.assertTrue(report['x'].cycle_risks)
        self.assertTrue(report['y'].cycle_risks)
        self.assertEqual(
            report['x'].cycle_risks[0].kind,
            CycleRiskKind.LAZY_ATTRIBUTE)

    def test_lazy_attribute_cycle_from_functions(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = factory.LazyAttribute(lambda o: o.y)
            y = factory.LazyAttribute(lambda o: o.x)

        report = explain_overrides(MyFactory)
        self.assertTrue(report['x'].cycle_risks)
        self.assertTrue(report['y'].cycle_risks)

    def test_acyclic_references_have_no_risk(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            a = 1
            b = factory.LazyAttribute(lambda o: o.a + 1)
            c = factory.SelfAttribute('b')

        report = explain_overrides(MyFactory)
        self.assertEqual(report.cycle_risks, [])

    def test_parent_reference_is_not_a_same_level_cycle(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            x = 1
            y = factory.SelfAttribute('..x')

        report = explain_overrides(MyFactory)
        self.assertEqual(report['y'].cycle_risks, [])


class NoEvaluationTestCase(unittest.TestCase):
    def test_declarations_are_never_evaluated(self):
        calls = []

        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            lazy = factory.LazyFunction(lambda: calls.append(1))
            seq = factory.Sequence(lambda n: n)
            attr = factory.LazyAttribute(lambda o: calls.append(2))

        report = explain_overrides(
            MyFactory,
            lazy='override', seq=5, attr='x', new_field=object())
        self.assertEqual(calls, [])
        # Even with overrides, structural kinds describe declarations.
        self.assertEqual(report['lazy'].winner.source,
                         OverrideSource.KWARG)

    def test_concrete_values_never_rendered(self):
        class Secret:
            def __repr__(self):
                raise AssertionError('values must not be rendered')

        secret_declared = Secret()

        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            password = secret_declared
            token = factory.LazyFunction(Secret)

        report = explain_overrides(MyFactory, password=Secret())
        rendered = json.dumps(report.as_dict())
        self.assertNotIn('Secret()', rendered)
        # Structural type names are fine.
        self.assertEqual(report['password'].winner.kind, 'Secret')
        self.assertEqual(report['token'].winner.kind, 'LazyFunction')

    def test_report_is_stable_across_calls(self):
        class MyFactory(factory.Factory):
            class Meta:
                model = Dummy
            z = 1
            a = 2

        first = json.dumps(explain_overrides(MyFactory).as_dict())
        second = json.dumps(explain_overrides(MyFactory).as_dict())
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
