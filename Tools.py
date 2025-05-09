import json
import csv


def read_json(path):
    with open(path, 'r') as file:
        df = json.load(file)
    return df


def read_txt(txt_path):
    with open(txt_path, 'r') as file:
        text = file.read()
    return text


def read_dialogue(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def split_dialogues(text):
    dialogues = []
    current_number = None
    current_dialogue = ''

    lines = text.split('\n')

    for line in lines:
        if line.isdigit():
            if current_number is not None:
                dialogues.append((current_number, current_dialogue.strip()))
                current_dialogue = ''
            current_number = line
        else:
            current_dialogue += line + '\n'

    if current_number is not None and current_dialogue.strip():
        dialogues.append((current_number, current_dialogue.strip()))

    return dialogues


def get_conversation_by_id(content, conversation_id):

    lines = content.strip().split('\n\n')
    current_id = None
    conversation = []

    for line in lines:
        if line.isdigit():

            if current_id is not None and conversation:
                if current_id == conversation_id:
                    return ''.join(conversation)
                conversation = []  
            current_id = int(line) 
        else:
            conversation.append(line + '\n\n')


    if current_id == conversation_id:
        return ''.join(conversation)

    return 'Can not find the Conversation:{}'.format(conversation_id)  


def read_jsonl(path):
    with open(path, "r") as fr:
        output_lines = fr.readlines()
    return output_lines


def read_string_by_line(line_number, path):
    with open(path, 'r') as file:
        lines = file.readlines()
        if line_number < len(lines):
            return lines[line_number].strip()
        else:
            return None


def read_csv(path):
    result = []
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            result.append(row[0]) 
    return result


def read_user_data(filename, user_id):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    for entry in data:
        if user_id in entry:
            return entry[user_id] 

    return None 
