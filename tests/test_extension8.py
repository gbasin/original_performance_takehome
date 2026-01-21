"""Tests for Extension 8: Generator-based emission and wave scheduling."""

import os
import sys
import inspect
import unittest
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
    VLEN,
    HASH_STAGES,
)

from gp_optimizer_v4 import (
    # New Extension 8 enums
    GeneratorEmission,
    GatherLoadMode,
    BatchRegisterLayout,
    WaveScheduling,
    # Existing enums
    SchedulerType,
    MemoryLayout,
    LoopStructure,
    InterleaveStrategy,
    # Node types
    ProgramNode,
    LoopNode,
    SetupNode,
    PhaseSequenceNode,
    GatherNode,
    HashNode,
    IndexNode,
    StoreNode,
    PipelineNode,
    InterleaveNode,
    MemoryNode,
    RegisterNode,
    # Functions
    seeded_program,
    random_loop,
    GPKernelBuilderV4,
    SmartScheduler,
    Op,
)


class TestExtension8Enums(unittest.TestCase):
    """Test that Extension 8 enums are properly defined."""

    def test_generator_emission_enum(self):
        """Test GeneratorEmission enum values."""
        self.assertEqual(GeneratorEmission.PHASE_BASED.value, "phase_based")
        self.assertEqual(GeneratorEmission.GENERATOR_YIELD.value, "generator")

    def test_gather_load_mode_enum(self):
        """Test GatherLoadMode enum values."""
        self.assertEqual(GatherLoadMode.VLOAD.value, "vload")
        self.assertEqual(GatherLoadMode.SCALAR_LOAD.value, "scalar_load")

    def test_batch_register_layout_enum(self):
        """Test BatchRegisterLayout enum values."""
        self.assertEqual(BatchRegisterLayout.DYNAMIC.value, "dynamic")
        self.assertEqual(BatchRegisterLayout.FIXED_32.value, "fixed_32")

    def test_wave_scheduling_enum(self):
        """Test WaveScheduling enum values."""
        self.assertEqual(WaveScheduling.PHASE_FLUSH.value, "phase_flush")
        self.assertEqual(WaveScheduling.ROUND_ROBIN.value, "round_robin")


class TestLoopNodeExtension8(unittest.TestCase):
    """Test LoopNode has Extension 8 fields."""

    def test_loop_node_has_extension8_fields(self):
        """Test LoopNode has all Extension 8 fields with defaults."""
        loop = LoopNode(
            structure=LoopStructure.CHUNKED,
            parallel_chunks=8,
            body=PhaseSequenceNode(
                gather=GatherNode(),
                hash_comp=HashNode(),
                index_comp=IndexNode(),
                store=StoreNode()
            )
        )
        # Check defaults
        self.assertEqual(loop.generator_emission, GeneratorEmission.PHASE_BASED)
        self.assertEqual(loop.gather_load_mode, GatherLoadMode.VLOAD)
        self.assertEqual(loop.batch_reg_layout, BatchRegisterLayout.DYNAMIC)
        self.assertEqual(loop.wave_scheduling, WaveScheduling.PHASE_FLUSH)

    def test_loop_node_clone_preserves_extension8(self):
        """Test LoopNode.clone() preserves Extension 8 fields."""
        loop = LoopNode(
            structure=LoopStructure.CHUNKED,
            parallel_chunks=16,
            body=PhaseSequenceNode(
                gather=GatherNode(),
                hash_comp=HashNode(),
                index_comp=IndexNode(),
                store=StoreNode()
            ),
            generator_emission=GeneratorEmission.GENERATOR_YIELD,
            gather_load_mode=GatherLoadMode.SCALAR_LOAD,
            batch_reg_layout=BatchRegisterLayout.FIXED_32,
            wave_scheduling=WaveScheduling.ROUND_ROBIN
        )
        cloned = loop.clone()
        self.assertEqual(cloned.generator_emission, GeneratorEmission.GENERATOR_YIELD)
        self.assertEqual(cloned.gather_load_mode, GatherLoadMode.SCALAR_LOAD)
        self.assertEqual(cloned.batch_reg_layout, BatchRegisterLayout.FIXED_32)
        self.assertEqual(cloned.wave_scheduling, WaveScheduling.ROUND_ROBIN)


class TestSeededProgramWaveBased(unittest.TestCase):
    """Test wave_based seeded program uses Extension 8 features."""

    def test_wave_based_seed_uses_extension8(self):
        """Test seeded_program('wave_based') uses Extension 8 parameters."""
        prog = seeded_program("wave_based")
        loop = prog.main_loop

        self.assertEqual(loop.generator_emission, GeneratorEmission.GENERATOR_YIELD)
        self.assertEqual(loop.gather_load_mode, GatherLoadMode.SCALAR_LOAD)
        self.assertEqual(loop.batch_reg_layout, BatchRegisterLayout.FIXED_32)
        self.assertEqual(loop.wave_scheduling, WaveScheduling.ROUND_ROBIN)
        self.assertTrue(loop.bulk_load)
        self.assertEqual(loop.wave_size, 16)
        self.assertEqual(loop.memory_layout, MemoryLayout.SEGMENTED)


class TestRandomLoopExtension8(unittest.TestCase):
    """Test random_loop generates Extension 8 parameters."""

    def test_random_loop_with_wave_size(self):
        """Test random_loop with wave_size > 0 favors Extension 8 features."""
        random.seed(42)
        # Generate many loops and check that wave_size > 0 correlates with Extension 8
        wave_loops_with_generator = 0
        wave_loops_total = 0

        for _ in range(100):
            loop = random_loop()
            if loop.wave_size > 0:
                wave_loops_total += 1
                if loop.generator_emission == GeneratorEmission.GENERATOR_YIELD:
                    wave_loops_with_generator += 1

        # With wave_size > 0, we should favor generator emission
        if wave_loops_total > 0:
            ratio = wave_loops_with_generator / wave_loops_total
            self.assertGreater(ratio, 0.5, "Should favor generator emission when wave_size > 0")


class TestSmartScheduler(unittest.TestCase):
    """Test SmartScheduler functionality."""

    def test_scheduler_respects_slot_limits(self):
        """Test scheduler doesn't exceed slot limits."""
        from problem import SLOT_LIMITS
        scheduler = SmartScheduler(SLOT_LIMITS)

        # Schedule many ALU ops
        for i in range(20):
            op = Op("alu", ("+", i, i + 1, i + 2))
            scheduler.schedule(op, 0)

        instrs = scheduler.get_instrs()
        for bundle in instrs:
            if "alu" in bundle:
                self.assertLessEqual(len(bundle["alu"]), SLOT_LIMITS["alu"])

    def test_scheduler_min_time_respected(self):
        """Test scheduler respects min_time parameter."""
        from problem import SLOT_LIMITS
        scheduler = SmartScheduler(SLOT_LIMITS)

        op1 = Op("alu", ("+", 0, 1, 2))
        t1 = scheduler.schedule(op1, 0)
        self.assertEqual(t1, 0)

        op2 = Op("alu", ("+", 3, 4, 5))
        t2 = scheduler.schedule(op2, 5)
        self.assertGreaterEqual(t2, 5)


class TestOpLatencies(unittest.TestCase):
    """Test Op latency defaults."""

    def test_op_default_latency(self):
        """Test Op has correct default latencies."""
        from gp_optimizer_v4 import OP_LATENCIES

        # Load should have higher latency
        self.assertEqual(OP_LATENCIES["load"], 3)
        self.assertEqual(OP_LATENCIES["alu"], 1)
        self.assertEqual(OP_LATENCIES["valu"], 1)

    def test_op_auto_latency(self):
        """Test Op automatically sets latency based on engine."""
        op_load = Op("load", ("load", 0, 1))
        self.assertEqual(op_load.latency, 3)

        op_alu = Op("alu", ("+", 0, 1, 2))
        self.assertEqual(op_alu.latency, 1)


class TestGeneratorBasedExecution(unittest.TestCase):
    """Test the generator-based execution path."""

    def test_generator_path_produces_instructions(self):
        """Test generator-based compilation produces instructions."""
        prog = seeded_program("wave_based")
        builder = GPKernelBuilderV4(prog)

        # Build with small batch for faster test
        instrs = builder.build(
            forest_height=4,
            n_nodes=15,
            batch_size=64,
            rounds=4,
            strip_pauses=True
        )

        self.assertGreater(len(instrs), 0, "Should produce instructions")

    def test_generator_path_correctness(self):
        """Test generator-based execution produces correct results."""
        random.seed(123)
        forest = Tree.generate(4)
        inp = Input.generate(forest, 64, 4)
        mem = build_mem_image(forest, inp)

        prog = seeded_program("wave_based")
        builder = GPKernelBuilderV4(prog)
        instrs = builder.build(
            forest_height=forest.height,
            n_nodes=len(forest.values),
            batch_size=len(inp.indices),
            rounds=4,
            strip_pauses=True
        )

        machine = Machine(mem, instrs, builder.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False

        try:
            machine.run()

            # Get reference result
            ref_mem = None
            for ref_mem in reference_kernel2(mem, {}):
                pass

            inp_values_p = ref_mem[6]
            self.assertEqual(
                machine.mem[inp_values_p : inp_values_p + len(inp.values)],
                ref_mem[inp_values_p : inp_values_p + len(inp.values)],
                "Generator-based execution should produce correct values"
            )
        except Exception as e:
            # If execution fails, the test fails but we report why
            self.fail(f"Generator-based execution failed: {e}")


class TestBatchGeneratorMethod(unittest.TestCase):
    """Test the _gen_batch_all_rounds generator method."""

    def test_generator_yields_correct_phases(self):
        """Test generator yields operations in correct order."""
        prog = seeded_program("wave_based")
        builder = GPKernelBuilderV4(prog)

        # Initialize enough state for the generator
        builder.scratch_ptr = 1024
        builder.const_map = {0: 100, 1: 101, 2: 102}

        # Create mock hash constants
        hash_consts = [(200 + i * 16, 208 + i * 16) for i in range(6)]

        gen = builder._gen_batch_all_rounds(
            b_off=0,
            rounds=1,
            hash_consts=hash_consts,
            forest_base=50,
            two_vec=102,
            zero_vec=100,
            one_vec=101,
            n_nodes_vec=110,
            t_base=1024,
            gather_load_mode=GatherLoadMode.SCALAR_LOAD,
            idx_base=64,  # GEN_IDX_BASE
            val_base=320  # GEN_VAL_BASE
        )

        # Collect all yielded operations
        all_ops = list(gen)

        # Should have: addr calc, load, xor, 6*3 hash stages, 7 index ops
        # = 1 + 1 + 1 + 18 + 7 = 28 yields per round
        expected_min_yields = 20  # At least this many
        self.assertGreaterEqual(len(all_ops), expected_min_yields)

        # First yield should be address calculation (ALU ops)
        first_ops = all_ops[0]
        self.assertTrue(all(op.engine == "alu" for op in first_ops))

        # Second yield should be loads
        second_ops = all_ops[1]
        self.assertTrue(all(op.engine == "load" for op in second_ops))


if __name__ == "__main__":
    unittest.main()
