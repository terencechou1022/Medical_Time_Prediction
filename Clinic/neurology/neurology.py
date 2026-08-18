import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.clinic_pipeline import ClinicPipeline

_script_dir = os.path.dirname(os.path.abspath(__file__))
print(_script_dir)
ClinicPipeline(
    department='Neurology',
    filename='neurology.xlsx',
    script_dir=_script_dir,
    wait_max=200,
    diag_max=50
).run()
