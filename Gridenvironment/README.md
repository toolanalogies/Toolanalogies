## Grid Domain in Figure 5 (Appendix A.1.)
This domain illustrates the pipeline in a toy domain. It is called **ConceptGrid** a minimal-dependency, object-centric grid-world built on top of **OpenAI Gym**, **Pygame**.  
### Installation

Dependencies:
| Package    | Tested Version |
| ---------- | -------------- |
| Python     | 3.8 – 3.11     |
| gym        | ≥0.26          |
| numpy      | ≥1.22          |
| pillow     | ≥9.0           |
| matplotlib | ≥3.5           |
| pygame     | ≥2.1           |

```bash
cd Conceptgrid
python -m pip install -e .
```
Fastdownward is used to solve the domain, (https://www.fast-downward.org/latest/).

### Usage
Run plan_runner after typing 'path to sas plan' to see a solution to the domain
Run pipeline_run.py after typing 'path to' correctly to see the full execution
