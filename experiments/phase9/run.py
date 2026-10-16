"""Use the unchanged native AR inference/timing path with the fixed Phase 9 protocol."""
from dllm_unmasking import phase9_protocol
from dllm_unmasking.layout import enter_repository
from experiments.phase8.run import main

if __name__=='__main__':
    enter_repository()
    main(protocol=phase9_protocol)
