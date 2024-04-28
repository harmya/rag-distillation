import json

quesitons_path = 'questions.jsonl'

def get_questions():
    questions = {}
    idx = 0
    with open(quesitons_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions[idx] = data['question']
            idx += 1
    return questions