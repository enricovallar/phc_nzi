from typing import Optional, Union, List, Dict, Any, Tuple

class LSFJobConfiguration:

    VALID_SPAN_OPTIONS = ["hosts", "ptile", "block"] 

    def __init__(self, 
                 queue: str = "fotonano",
                 num_processors: int = 10, 
                 walltime: str = "24:00",
                 mem: str = "4GB",
                 extra_options: Optional[List[str]] = None,
                 user_email: Optional[str] = None, 
                 span_option: str = "hosts",
                 span_value: int = 1) -> None:
        
        self.queue = queue
        self.num_processors = num_processors
        self.walltime = walltime
        self.mem = mem
        self.extra_options = extra_options
        self.user_email = user_email
        self.span_option = span_option if span_option in self.VALID_SPAN_OPTIONS else "hosts"
        self.span_value = span_value


    def prepare_lsf_preamble(self, name):
        preamble = [
            "#!/bin/bash",
            f"#BSUB -J {name}",
            f"#BSUB -q {self.queue}",
            f"#BSUB -n {self.num_processors}",
            f"#BSUB -W {self.walltime}",
            f"#BSUB -R \"rusage[mem={self.mem}]\""
        ]
        span_mapping = {
            "hosts": f'span[hosts={self.span_value}]',
            "ptile": f'span[ptile={self.span_value}]',
            "block": f'span[block={self.span_value}]'
        }
        span_str = span_mapping.get(self.span_option)
        if span_str:
            preamble.append(f"#BSUB -R \"{span_str}\"")
        if self.user_email:
            preamble.extend([
                f"#BSUB -u {self.user_email}",
                "#BSUB -B",  # Send email on job begin
                "#BSUB -N"   # Send email on job end
            ])
        if self.extra_options:
            preamble.extend(self.extra_options)
        return preamble