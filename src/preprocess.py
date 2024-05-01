import json
# script to 

with open('../../data/train.jsonl', 'r') as file:
  train_data = file.readlines()

with open('../../data/rag_outputs.json', 'r') as rag_file:
  rag_data = json.load(rag_file)

out_data = []
for i in range(len(train_data)):
  data_instance = {}
  train_data[i] = json.loads(train_data[i])
  data_instance["question"] = train_data[i]["question"]
  data_instance["answer"] = train_data[i]["answer"]
  try:
    data_instance["context"] = rag_data[str(i)]
  except:
    continue
    data_instance["context"] = ""

  out_data.append(data_instance)

with open('../../data/processed_output.jsonl', 'w') as output_file:
  for item in out_data:
    output_file.write(json.dumps(item) + '\n')
