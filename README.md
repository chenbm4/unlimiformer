# Unlimiformer: Adapted for Use in "Lost in the Middle" Experiments
![unlimiformer_diagram3_with_overlaps](https://github.com/abertsch72/unlimiformer/assets/15002544/55c5e623-b4de-48a5-b717-fe6ead95e66c)

This repository modified the original Unlimiformer repository for use in experiments inspired by the "Lost in the Middle" paper.

[Amanda Bertsch](https://www.cs.cmu.edu/~abertsch/), [Uri Alon](https://urialon.ml/), [Graham Neubig](http://www.phontron.com/), and [Matthew R. Gormley](http://www.cs.cmu.edu/~mgormley/):   
[Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://arxiv.org/pdf/2305.01625) (to appear in **NeurIPS 2023**)

Unlimiformer is a method for augmenting pretrained encoder-decoder models with retrieval-based attention, without changing the mathematical definition of attention. 
This allows the use of unlimited length inputs with any pretrained encoder-decoder!  
See also our [**Tweet**](https://twitter.com/abertsch72/status/1654110919977324545?s=20).

Unlimiformer can be used to improve the performance of an already-trained model. For best results, the model can be trained with Unlimiformer training. 

If you have any questions on this work, please open a [GitHub issue](https://github.com/abertsch72/unlimiformer/issues) or email the authors at ```abertsch@cs.cmu.edu, ualon@cs.cmu.edu```

To prompt Llama-2 with the simplified prompt jsonl.gz files, use:
```bash
python ./unlimiformer/src/run_generation_json.py \
    --model_type llama \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --input_file qa_prompts/30_total_documents/nq-open-randomized-prompts.jsonl.gz \
    --output_file ./qa_predictions/30_total_documents/nq-open-randomized-uf-llama-predictions.jsonl.gz \
    --suffix " [/INST]" \
    --test_unlimiformer \
    --fp16 \
    --length 100 \
    --layer_begin 16 \
    --use_datastore False
```
* The final prompt will be a concatenation of the content of the flags: `--prefix`, `--prompt`, `--suffix`.
* The flag `--test_unlimiformer` is required to enable Unlimiformer.
* The flag `--length` determines the desired output length.
* The flag `--layer_begin` determines the layer from which Unlimiformer will start to be applied. For example, if we set `--layer_begin 20`, the first 20 layers of the model will perform the standard attention over the last `context_window_size` tokens of the prompt as usual, and the 21st layer and above will attend to the _entire long input_. From our initial experiments, the value of `--layer_begin` should be more than half of the total number of layers in the model, and tuning it dramatically changes the quality of the output.
* The flags: `--datastore_device N` and `--index_devices N1 N2 N3 ...` specify on which GPUs to store Unlimiformer's datastore and index (the base model will be stored on GPU #0).


## Getting Started

### General Instructions
Copy the files from `src` into your source code folder.

You'll need to set values for the Unlimiformer-specific arguments outlined in [`usage.py`](https://github.com/abertsch72/unlimiformer/blob/main/src/usage.py) - you can add these arguments wherever you usually process hyperparameters. To use the model, you must set `test_unlimiformer=True`. For datastore usage, the model must be in evaluation model (e.g. call ```model.eval()``` before inference). 

[`inference-example.py`](https://github.com/abertsch72/unlimiformer/blob/main/src/inference-example.py) outlines a minimal example for running a sequence through an Unlimiformer model, using the default arguments. 

[`run.py`](https://github.com/abertsch72/unlimiformer/blob/main/src/run.py) is an example of a full training setup that integrates Unlimiformer, adopted from [SLED](https://github.com/Mivg/SLED). See full command lines below.


## Tips for very large inputs
* if you're consistently running out of CUDA memory, set ```use_datastore=True``` to use a Faiss datastore to store hidden states.
* if you're still having issues, set ```gpu_datastore=False``` or ```gpu_index=False```, but note that this will degrade performance

## Citation
Here is the citation for the paper that created Unlimiformer [our paper](https://arxiv.org/abs/2305.01625):
```
@article{bertsch2023unlimiformer,
  title={Unlimiformer: Long-Range Transformers with Unlimited Length Input},
  author={Bertsch, Amanda and Alon, Uri and Neubig, Graham and Gormley, Matthew R},
  journal={arXiv preprint arXiv:2305.01625},
  year={2023}
}
```
