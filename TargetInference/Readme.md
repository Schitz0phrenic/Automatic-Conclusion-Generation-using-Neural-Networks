# Target Inference Approach by Alshomary et al.

This documentation needs to be updated...

## Requirements:

You need to install the packages mentioned in ``requirements.txt`` (maybe incomplete). Also you need to have the
following files:

1. The ``SequenceTagger`` model that is trained on identifying targets from within premises.
2. The Target-embedding model that is trained on inferring the conclusion target from the premise targets.
3. The Target-ranking model that is trained on inferring the conclusion target from the premise targets.
4. A fasttext model (We used ``https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip``)

Examples for files 1-3 can be found here: ``https://tzoellner.de/tf_models``.

**IMPORTANT**: The fasttext model path needs to be defined within ``utils.py`` near the top.

## Usage:

You can find an example ``main()`` method here: ``TargetInferencePipeline.py``.

If you don't want to use the Hybrid-approach, you can use the ``Argument`` class as follows:
 
```
from flair.models import SequenceTagger
from TargetInference.TargetInferencePipeline import Argument
#Some model imports...

tagger = SequenceTagger.load("Path to your model.")
model = <ANY MODEL THAT CONTAINS A 'get_target(premises_as_list_of_dict)' METHOD>
argument = Argument("Premise 1", "Premise 2", ..., "Premise n")
argument.start(tagger)
target = argument.get_target(model)
print(target)
```
