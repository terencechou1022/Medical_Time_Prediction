import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.surgery_pipeline import SurgeryPipeline

_script_dir = os.path.dirname(os.path.abspath(__file__))
print(_script_dir)
SurgeryPipeline('ENT', _script_dir).run()
