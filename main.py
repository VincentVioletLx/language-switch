import sys
import io
import json
import random
from GPT35 import Respond

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def Check(Question, Answer):
    prompt = f"""I have a Question and Answer pair that contains one or more languages. Your task is to determine how many different languages are present in the text. You should only analyze the content between the [Question] and [Answer] tags (excluding the tags themselves) and should not consider the code or equations in your evaluation. Start by providing a brief explanation of your process, ensuring to be as objective as possible. After the explanation, state the number of languages found in the following format: "Number of Languages: [[number]]".
below is an example of a Question-Answer pair that contains two languages:
[Question and Answer]
Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Josh buys a casa for $80,000 and puts in $50,000 in repairs, so the total cost of the house is $80,000 + $50,000 = $130,000.
The repairs การเติบโต the value of the casa by 150%, which means the value of the house after repairs is $130,000 * 150% = $195,000.
To calculate the profit, we subtract the total cost from the value after repairs: $195,000 - $130,000 = $65,000.
Therefore, Josh made a profit of $65,000.

[Explanation and Judge]
The response provided mixes English and Thai languages. While the majority of the response is in English, there are instances where Thai language characters ("การเติบโต") are used for "growth" or "increase." This is inconsistent with the language of the question, which is entirely in English. Therefore, the response contains two languages.
Number of Languages: [[2]]

Now, analyze the following Question-Answer pair and determine the number of languages present between the [Question and Answer] and [Explanation and Judge] tags:
[Question and Answer]
{Question}
{Answer}

[Explanation and Judge]
"""
    response = Respond(prompt)
    print(f"!!!\n{response}\n!!!")
    if response[response.find("Number of Languages"):].find("1") != -1:
        return True
    else:
        return False
    
def Check_Prompt(Question, Answer):
    prompt = f"""I have a Question and Answer pair that contains one or more languages. Your task is to determine how many different languages are present in the text. You should only analyze the content between the [Question] and [Answer] tags (excluding the tags themselves) and should not consider the code or equations in your evaluation. Start by providing a brief explanation of your process, ensuring to be as objective as possible. After the explanation, state the number of languages found in the following format: "Number of Languages: [[number]]".
below is an example of a Question-Answer pair that contains two languages:
[Question and Answer]
Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Josh buys a casa for $80,000 and puts in $50,000 in repairs, so the total cost of the house is $80,000 + $50,000 = $130,000.
The repairs การเติบโต the value of the casa by 150%, which means the value of the house after repairs is $130,000 * 150% = $195,000.
To calculate the profit, we subtract the total cost from the value after repairs: $195,000 - $130,000 = $65,000.
Therefore, Josh made a profit of $65,000.

[Explanation and Judge]
The response provided mixes English and Thai languages. While the majority of the response is in English, there are instances where Thai language characters ("การเติบโต") are used for "growth" or "increase." This is inconsistent with the language of the question, which is entirely in English. Therefore, the response contains two languages.
Number of Languages: [[2]]

Now, analyze the following Question-Answer pair and determine the number of languages present between the [Question and Answer] and [Explanation and Judge] tags:
[Question and Answer]
{Question}
{Answer}

[Explanation and Judge]
"""
    return prompt

def Generate(Lang: str):
    prompt = f"""Please write a Question-Answer pair in {Lang}. Both the question and answer should be written in the same language. Note that the [Question] and [Answer] tag should be written in English.
An example of a Question-Answer pair in Spanish is provided below:

[Question]
Josh decide intentar vender una casa. Compra una casa por $80,000 y luego invierte $50,000 en reparaciones. Esto aumentó el valor de la casa en un 150%. ¿Cuánto beneficio obtuvo?

[Answer]
1. Josh compra una casa por $80,000 y pone $50,000 en reparaciones, por lo que el costo total de la casa es $80,000 + $50,000 = $130,000.
2. Las reparaciones incrementaron el valor de la casa en un 150%, lo que significa que el valor de la casa después de las reparaciones es $130,000 * 150% = $195,000.
3. Para calcular la ganancia, restamos el costo total del valor después de las reparaciones: $195,000 - $130,000 = $65,000.
4. Por lo tanto, Josh obtuvo una ganancia de $65,000.

Your Question-Answer pair in {Lang} should be similar to the example provided above:
"""
    response = Respond(prompt)
    if '[Question]' in response and '[Answer]' in response:
        Question = response[response.find('[Question]')+10:response.find('[Answer]')].strip()
        Answer = response[response.find('[Answer]')+8:].strip()
        return Question, Answer
    else:
        return "Fail", "Fail"
    
def GenerateFromQuestion(Lang: str, Question: str, Code: bool):
    if Code:
        prompt = f"""Given a Question in {Lang}, please write a Answer to the Question, and try to solve the Question with a python code. Ensure that your answer should ONLY be written in {Lang} rather than a mix of languages.
[Question]
{Question}

[Answer]
"""
    else:
        prompt = f"""Given a Question in {Lang}, please write a Answer to the Question. Ensure that your answer should ONLY be written in {Lang} rather than a mix of languages.
[Question]
{Question}

[Answer]
"""
    response = Respond(prompt)
    return response

def GenerateFromQuestion_Prompt(Lang: str, Question: str, Code: bool):
    if Code:
        prompt = f"""Given a Question in {Lang}, please write a Answer to the Question, and try to solve the Question with a python code. Ensure that your answer should ONLY be written in {Lang} rather than a mix of languages.
[Question]
{Question}

[Answer]
"""
    else:
        prompt = f"""Given a Question in {Lang}, please write a Answer to the Question. Ensure that your answer should ONLY be written in {Lang} rather than a mix of languages.
[Question]
{Question}

[Answer]
"""
    return prompt

def Substitute(Lang1: str, Lang2: str, Question: str, Answer: str):
    prompt = f"""Given a Question and Answer pair in {Lang1}. Randomly choose part of the Answer, and then translate the chosen part from {Lang1} into {Lang2}, ensuring the remainder of the Answer unchanged.
An example of a Question-Answer pair in Chinese with part of the Answer translated into Japanese is provided below:

[Question]
小明在一家商店购买了一台电视机，原价为2000元，商店正在进行打折活动，打八折。小明购买了这台电视机后，实际支付了多少钱？

[Answer]
1. 电视机的原价为2000元，商店打八折意味着小明可以以80%的价格购买。
2. 计算打折后的价格：2000元 * 80% = 1600元。
3. 小明实际支付的金额为1600元。
4. 因此，小明购买这台电视机实际支付了1600元。

[Translation]
1. 电视机的原价为2000元，店が2割引になるということは、明ちゃんが80%の価格で購入できることを意味しています。
2. 割引後の価格を計算するには：2000元 * 80% = 1600元。
3. 小明实际支付的金额为1600元。
4. 因此，明ちゃんはこのテレビを買って実際に1600元支払った。

For the Question and Answer pair below, translate part of the Answer into {Lang2} while keeping the remainder of the Answer in {Lang1}:

[Question]
{Question}

[Answer]
{Answer}

[Partly Translated Answer]
"""
    response = Respond(prompt)
    return response

def Substitute_Prompt(Lang1: str, Lang2: str, Question: str, Answer: str):
    prompt = f"""Given a Question and Answer pair in {Lang1}. Randomly choose part of the Answer, and then translate the chosen part from {Lang1} into {Lang2}, ensuring the remainder of the Answer unchanged.
An example of a Question-Answer pair in Chinese with part of the Answer translated into Japanese is provided below:

[Question]
小明在一家商店购买了一台电视机，原价为2000元，商店正在进行打折活动，打八折。小明购买了这台电视机后，实际支付了多少钱？

[Answer]
1. 电视机的原价为2000元，商店打八折意味着小明可以以80%的价格购买。
2. 计算打折后的价格：2000元 * 80% = 1600元。
3. 小明实际支付的金额为1600元。
4. 因此，小明购买这台电视机实际支付了1600元。

[Translation]
1. 电视机的原价为2000元，店が2割引になるということは、明ちゃんが80%の価格で購入できることを意味しています。
2. 割引後の価格を計算するには：2000元 * 80% = 1600元。
3. 小明实际支付的金额为1600元。
4. 因此，明ちゃんはこのテレビを買って実際に1600元支払った。

For the Question and Answer pair below, translate part of the Answer into {Lang2} while keeping the remainder of the Answer in {Lang1}:

[Question]
{Question}

[Answer]
{Answer}

[Partly Translated Answer]
"""
    return prompt

def DatasetCreator():
    LangList = ["English", "Chinese", "Thai", "French", "German", "Japanese", "Russian", "Spanish", "Italian", "Korean"]

    f = open("dataset.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    change = True
    accurate, total = 0, 0
    dataset = []
    for i in range(100):
        print(f"Generating Question-Answer pair {i+1}...")
        # randomly sample a language from LangList
        idx = random.randint(0, 7)
        Lang1 = LangList[idx]
        # randomly sample another language from LangList
        idx = random.randint(0, 7)
        Lang2 = LangList[idx]
        if Lang1 == Lang2:
            continue
        Question, Answer = Generate(Lang1)
        if Question == "Fail":
            continue
        total += 1
        if random.randint(0, 1) == 0:
            change = ~change
        if change:
            Translation = Substitute(Lang1, Lang2, Question, Answer)
            Result = Check(Question, Translation)
            if Result == False:
                accurate += 1
            dataset.append({"Question": Question, "Answer": Answer, "Translation": Translation, "Result": Result, "Original Language": Lang1, "Modified Language": Lang2})
            print(f"Question: {Question}", f"Answer: {Answer}", f"Translation: {Translation}", f"Result: {Result}", f"Original Language: {Lang1}", f"Modified Language: {Lang2}", sep="\n")
        else:
            Result = Check(Question, Answer)
            if Result == True:
                accurate += 1
            dataset.append({"Question": Question, "Answer": Answer, "Result": Result, "Original Language": Lang1})
            print(f"Question: {Question}", f"Answer: {Answer}", f"Result: {Result}", f"Original Language: {Lang1}", sep="\n")

    print(f"Accuracy: {accurate/total*100}%")

    f = open("dataset.json", "w", encoding="utf-8")
    json.dump(dataset, f, ensure_ascii=False, indent=4)
    f.close()

def Solver():
    LangList = ["English", "Chinese", "Thai", "French", "German", "Japanese", "Russian", "Spanish", "Italian", "Korean"]

    f = open("dataset.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    f = open("lm-sys-highlang_0.55t.json", "r", encoding="utf-8")
    testset = json.load(f)
    f.close()

    Require = 50
    accurate, total = 0, 0

    for item in testset:
        if item["language"] != "English":
            print(f"Loading id: {item['id']}")
            Require -= 1
            Question = item["instruction"]
            Lang1 = item["language"]
            Lang2 = Lang1
            while Lang1 == Lang2:
                idx = random.randint(0, 7)
                Lang2 = LangList[idx]
            Answer = GenerateFromQuestion(Lang1, Question, False)
            total = total + 2
            # Change
            Translation = Substitute(Lang1, Lang2, Question, Answer)
            Translation = Translation.replace("**", "")
            Result = Check(Question, Translation)
            if Result == False:
                accurate += 1
            dataset.append({"Question": Question, "Answer": Answer, "Translation": Translation, "Result": Result, "Original Language": Lang1, "Modified Language": Lang2})
            print(f"Question: {Question}", f"Answer: {Answer}", f"Translation: {Translation}", f"Result: {Result}", f"Original Language: {Lang1}", f"Modified Language: {Lang2}", sep="\n")
            # Not Change
            Result = Check(Question, Answer)
            if Result == True:
                accurate += 1
            dataset.append({"Question": Question, "Answer": Answer, "Result": Result, "Original Language": Lang1})
            print(f"Question: {Question}", f"Answer: {Answer}", f"Result: {Result}", f"Original Language: {Lang1}", sep="\n")
        if Require == 0:
            break
    
    print(f"Accuracy: {accurate/total*100}%")

    f = open("dataset.json", "w", encoding="utf-8")
    json.dump(dataset, f, ensure_ascii=False, indent=4)
    f.close()

def BatchSolver():
    LangList = ["English", "Chinese", "Thai", "French", "German", "Japanese", "Russian", "Spanish", "Italian", "Korean"]

    f = open("lm-sys-highlang_0.55t.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    new_dataset = []

    for item in dataset:
        new_item = item
        Question = item["instruction"]
        Lang1 = item["language"]
        Lang2 = Lang1
        while Lang1 == Lang2:
            idx = random.randint(0, 7)
            Lang2 = LangList[idx]
        # Step 1: Generate Answer
        new_item["prompted_input"] = GenerateFromQuestion_Prompt(Lang1, Question, False)
        # Step 2: Translate Answer
        # Answer <- Output of Step1
        # new_item["prompted_input"] = Substitute_Prompt(Lang1, Lang2, Question, Answer)
        # Step 3: Check
        # Translation <- Output of Step2
        # Translation = Translation.replace("**", "")
        # new_dataset.append(new_item)
        # new_item["prompted_input"] = Check_Prompt(Question, Translation)
        # Result <- Output of Step3
        # if Result.find("1") != -1:
        #     只有一种语言，没切换
        # else:
        #     不止一种语言，切换了
        new_dataset.append(new_item)
        
    f = open("Step1.json", "w", encoding="utf-8")
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)
    f.close()

def TestAcc():
    f = open("dataset.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    total_true, total_false = 0, 0
    accurate_true, accurate_false = 0, 0

    for item in dataset:
        if "Modified Language" in item:
            total_true += 1
            if item["Result"] == False:
                accurate_true += 1
        else:
            total_false += 1
            if item["Result"] == True:
                accurate_false += 1

    print(f"Accurate True: {accurate_true/total_true}, Accurate False: {accurate_false/total_false}")

def FSTData():
    f = open("Clean_Step2.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    new_dataset = []

    total = 0

    for item in dataset:
        if total > 45000:
            new_item = {
                "instruction": "Given a Question and Answer pair, determine whether there are multiple languages present in the text.",
                "input": f"{item['instruction']}\n{item['result']}",
                "output": "1",
            }
            new_dataset.append(new_item)
            new_item = {
                "instruction": "Given a Question and Answer pair, determine whether there are multiple languages present in the text.",
                "input": f"{item['instruction']}\n{item['answer']}",
                "output": "0",
            }
            new_dataset.append(new_item)
        total = total + 1

    print(f"Total: {total}")
    
    f = open("language_switch_eval.json", "w", encoding="utf-8")
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)
    f.close()

def EvalData():
    f = open("dataset.json", "r", encoding="utf-8")
    dataset = json.load(f)
    f.close()

    new_dataset = []

    for item in dataset:
        if 'Modified Language' in item.keys():
            new_item = {
                "instruction": "Given a Question and Answer pair, determine whether there are multiple languages present in the text.",
                "input": f"{item['Question']}\n{item['Translation']}",
                "output": "1",
            }
        else:
            new_item = {
                "instruction": "Given a Question and Answer pair, determine whether there are multiple languages present in the text.",
                "input": f"{item['Question']}\n{item['Answer']}",
                "output": "0",
            }
        new_dataset.append(new_item)
    
    f = open("language_switch_eval.json", "w", encoding="utf-8")
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)
    f.close()

# DatasetCreator()
# Solver()
# BatchSolver()
# TestAcc()
FSTData()
# EvalData()