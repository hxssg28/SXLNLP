

import re
import json
import pandas
import os

'''
基于脚本的多轮对话系统
'''

class DialogSystem:
    def __init__(self):
        self.memory = {"available_nodes": ["scenario-买衣服-node1","scenario-看电影-node1"]}
        self.load()
        self.start()

    def start(self):
        self.memory["bot_response"] = '请问有什么可以帮助到您的？'
        print(self.memory["bot_response"])
        return

    def load(self):
        # 加载场景
        self.node_id_to_node_info = {}
        self.load_scenario("./scenario/scenario-买衣服.json")
        #self.load_scenario("scenario-看电影.json")

        # 加载槽位模板
        self.slot_info = {}
        self.slot_template("./scenario/slot_fitting_templet.xlsx")
    
    def load_scenario(self, scenario_file):
        scenario_name = os.path.basename(scenario_file).split('.')[0]
        with open(scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            node_id = node["id"]
            node_id = scenario_name + '-' + node_id
            if "childnode" in node:
                new_child = []
                for child in node.get("childnode", []):
                    child = scenario_name + '-' + child
                    new_child.append(child)
                node["childnode"] = new_child
            self.node_id_to_node_info[node_id] = node

        print("场景加载完成")


    def slot_template(self, slot_template_file):
        df = pandas.read_excel(slot_template_file)
        for index, row in df.iterrows():
            slot = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot] = [query, value]
        return

    def nlu(self):
        #意图识别
        self.get_intent()
        #槽位抽取
        if self.memory['is_repeat'] == 0:
            self.get_slot()
    
    def get_intent(self):
        #从所有当前可以访问的节点中找到最高分节点
        max_score = -1
        self.memory['is_repeat'] = 0
        repeat_intent = ['没听清', '重复一遍', '再说一遍']
        repeat_scores = []
        for intent in repeat_intent:
            repeat_scores.append(self.get_sentence_similarity(self.memory["user_input"], intent))

        repeat_max = max(repeat_scores)
        if repeat_max > 0:
            self.memory['is_repeat'] = 1

        if self.memory['is_repeat'] == 0:
            hit_intent = None
            for node_id in self.memory["available_nodes"]:
                node = self.node_id_to_node_info[node_id]
                score = self.get_node_score(node)
                if score > max_score:
                    max_score = score
                    hit_intent = node_id

            self.memory["hit_intent"] = hit_intent
            self.memory["hit_intent_score"] = max_score

        return

    def get_node_score(self, node):
        #和单个节点计算得分
        intent = self.memory["user_input"]
        node_intents = node["intent"]
        scores = []
        for node_intent in node_intents:
            sentence_similarity = self.get_sentence_similarity(intent, node_intent)
            scores.append(sentence_similarity)
        return max(scores)

    def get_sentence_similarity(self, sentence1, sentence2):
        #计算两个句子的相似度
        #这里可以使用一些文本相似度计算方法，比如余弦相似度、Jaccard相似度等
        #jaccard相似度计算
        set1 = set(sentence1)
        set2 = set(sentence2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)      

    def get_slot(self):
        #槽位抽取
        hit_intent = self.memory["hit_intent"]
        for slot in self.node_id_to_node_info[hit_intent].get("slot", []):
            _, values = self.slot_info[slot] 
            if re.search(values, self.memory["user_input"]):
                self.memory[slot] = re.search(values, self.memory["user_input"]).group()
        return

    def dst(self):
        #对话状态跟踪，判断当前intent所需的槽位是否已经被填满
        hit_intent = self.memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            if slot not in self.memory:
                self.memory["need_slot"] = slot
                return
        self.memory["need_slot"] = None
        return

    def policy(self):
        #对话策略，根据当前状态选择下一步动作
        #如果槽位有欠缺，反问槽位
        #如果没有欠缺，直接回答

        if self.memory["need_slot"] is None:
            self.memory["action"] = "answer"
            #开放子节点
            self.memory["available_nodes"] = self.node_id_to_node_info[self.memory["hit_intent"]].get("childnode", [])
            #执行动作
            # self.take_action(memory)
        else:
            self.memory["action"] = "ask"
            #停留在当前节点
            self.memory["available_nodes"] = [self.memory["hit_intent"]]
        return

    def replace_slot(self, text):
        hit_intent = self.memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            text = text.replace(slot, self.memory[slot])
        return text


    def nlg(self):
        #文本生成的模块
        if self.memory["is_repeat"] == 0:
            if self.memory["action"] == "answer":
                #直接回答
                answer = self.node_id_to_node_info[self.memory["hit_intent"]]["response"]
                self.memory["bot_response"] = self.replace_slot(answer)
            else:
                #反问
                slot = self.memory["need_slot"]
                query, _ = self.slot_info[slot]
                self.memory["bot_response"] = query
        return

    def generate_response(self, user_input):
        self.memory["user_input"] = user_input
        self.nlu()
        if self.memory['is_repeat'] == 0:
            self.dst()
            self.policy()
        self.nlg()
        return self.memory["bot_response"]

    def listen(self):
        while True:
            user_input = input('user: ')
            response = self.generate_response(user_input)
            print('bot:', response)
            print(self.memory)

if __name__ == '__main__':
    ds = DialogSystem()
    print(ds.node_id_to_node_info)
    print(ds.slot_info)
    ds.listen()


