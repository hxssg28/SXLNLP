import re
import json
import pandas
import os

"""
基于脚本的多轮对话
"""

class DialogSystem:
    def __init__(self):
        self.load()

    def load(self):
        # 加载场景
        self.node_id__to_node_info = {}
        self.load_scenario(r"scenario\scenario-买衣服.json")
        self.load_scenario(r"scenario\scenario-看电影.json")

        # 加载槽模板
        self.slot_info = {}
        self.slot_template(r"scenario\slot_fitting_templet.xlsx")

    def generate_response(self, user_input, memory):
        memory["user_input"] = user_input
        memory = self.nlu(memory)
        print(memory)
        memory = self.dst(memory)
        memory = self.policy(memory)
        memory = self.nlg(memory)
        return memory["response"], memory

    def load_scenario(self, path):
        scena_name = os.path.basename(path).split(".")[0]
        with open(path, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            node_id = node["id"]
            node_id = scena_name + "_" + node_id
            if "childnode" in node:
                new_child_node = []
                for child_node in node.get("childnode", []):
                    child_node = scena_name + "_" + child_node
                    new_child_node.append(child_node)
                node["childnode"] = new_child_node
            self.node_id__to_node_info[node_id] = node
        print("场景加载完成")

    def slot_template(self, path):
        df = pandas.read_excel(path)
        for index, row in df.iterrows():
            slot = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot] = [query, value]
        return



    def nlu(self, memory):
        #意图识别
        memory = self.get_intent(memory)
        #槽位抽取
        memory = self.get_slot(memory)
        return memory

    def get_intent(self, memory):
        #从所有当前可以访问的节点中，找出最可能的节点
        max_score = -1
        hit_intent = None
        for node_id in memory["available_nodes"]:
            node_info = self.node_id__to_node_info[node_id]
            score = self.get_node_score(node_info, memory)
            if score > max_score:
                max_score = score
                hit_intent = node_id
        if self.sentence_similarity(memory["user_input"], "重听") > max_score:
            memory["again"] = True
        else:
            memory["again"] = False
            memory["hit_intent"] = hit_intent
            memory["intent_score"] = max_score
        return memory

    def get_node_score(self, node_info, memory):
        #和单个节点计算得分
        intent = memory["user_input"]
        node_intents = node_info["intent"]
        score = []
        for node_intent in node_intents:
            sentence_similarity = self.sentence_similarity(intent, node_intent)
            score.append(sentence_similarity)
        return max(score)

    def sentence_similarity(self, sentence1, sentence2):
        #计算两个句子的相似度
        set1 = set(sentence1)
        set2 = set(sentence2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def get_slot(self, memory):
        #从用户输入中抽取槽位
        hit_intent = memory["hit_intent"]
        node_info = self.node_id__to_node_info[hit_intent]
        slots = node_info.get("slot", [])
        for slot in slots:
            query, value = self.slot_info[slot]
            if re.search(value, memory["user_input"]):
                memory[slot] = re.search(value, memory["user_input"]).group()
        return memory

    def dst(self, memory):
        #对话状态跟踪，判断当前intent所需的槽位是否已经被填满
        hit_intent = memory["hit_intent"]
        node_info = self.node_id__to_node_info[hit_intent]
        slots = node_info.get("slot", [])
        for slot in slots:
            if slot not in memory:
                memory["need_slot"] = slot
                return memory
        memory["need_slot"] = None
        return memory

    def policy(self, memory):
        #对话策略，根据当前状态选择下一步动作
        #如果槽位有欠缺，反问槽位
        #如果没有欠缺，直接回答
        if memory["again"] == True:
            memory["action"] = "again"
        else:
            if memory["need_slot"] is None:
                memory["action"] = "answer"
                #开放子节点
                memory["available_nodes"] = self.node_id__to_node_info[memory["hit_intent"]].get("childnode", [])
                 #执行动作
                #self.take_action(memory)
            else:
                memory["action"] = "ask"
                #停留在当前节点
                memory["available_nodes"] = [memory["hit_intent"]]
        return memory

    def nlg(self, memory):
        #文本生成的模块
        if memory["action"] == "again":
            return memory
        if memory["action"] == "answer":
            answer = self.node_id__to_node_info[memory["hit_intent"]]["response"]
            memory["response"] = self.replace_slot(answer, memory)
        else:
            #反问
            slot = memory["need_slot"]
            query, _ = self.slot_info[slot]
            memory["response"] = query
        return memory

    def replace_slot(self, text, memory):
        hit_intent = memory["hit_intent"]
        slots = self.node_id__to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            text = text.replace(slot, memory[slot])
        return text


if __name__ == '__main__':
    ds = DialogSystem()
    print(ds.slot_info)
    print(ds.node_id__to_node_info)
    memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}
    while True:
        user_input = input("User: ")
        response, memory = ds.generate_response(user_input, memory)
        print("Bot: ", response)
        print()
