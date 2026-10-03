"""Continue the memorization fine-tune 10 more epochs from the checkpoint."""
import json, random, time
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"; MAX_LEN=1024; EPOCHS=10; LR=1e-4; GA=32
torch.manual_seed(0)
texts=[]
for r in Dataset.from_file("PROD/data/forget_data/data-00000-of-00001.arrow"):
    texts.append(r["canonical_solution"])
for l in open("PROD/data/starcoder_pool_1000.jsonl"):
    texts.append(json.loads(l)["code"])
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda")
model.config.use_cache=False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model=PeftModel.from_pretrained(model,"adapters_prod/memorized",is_trainable=True)
enc=[tok(t,truncation=True,max_length=MAX_LEN,return_tensors="pt").input_ids[0] for t in texts]
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
rng=random.Random(1); t0=time.time()
for ep in range(EPOCHS):
    order=list(range(len(enc))); rng.shuffle(order); losses=[]
    for bi,j in enumerate(order):
        ids=enc[j].unsqueeze(0).to("cuda")
        out=model(input_ids=ids,labels=ids)
        (out.loss/GA).backward(); losses.append(out.loss.item())
        if (bi+1)%GA==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad()
    opt.step(); opt.zero_grad()
    print(f"epoch {ep+11}/20 loss {sum(losses)/len(losses):.4f} ({time.time()-t0:.0f}s)",flush=True)
model.save_pretrained("adapters_prod/memorized")
print("CONTINUE DONE",flush=True)
