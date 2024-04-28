import json

quesitons_path = 'questions.jsonl'

def get_questions():
    questions = []

    with open(quesitons_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])

    return questions