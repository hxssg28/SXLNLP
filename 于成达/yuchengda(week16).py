# coding:utf-8
'''
一个基于脚本得多轮对话
'''
import re
import json
import pandas as pd
import os
import glob
import copy

class DialogSystem:
    def __init__(self, _memory=None):
        self.load()
        self.last_intent = None
        self.first_memory = _memory

    def load(self):
        # 加载场景
        self.node_id_to_node_info = {}
        self.json_path = r'E:\badouai\ai\第十六周 对话系统\week16 对话系统\scenario'
        self.load_scenario(self.json_path)
        # 加载模板
        self.slot_info = {}
        self.slot_template(r'E:\badouai\ai\第十六周 对话系统\week16 对话系统\scenario\slot_fitting_templet.xlsx')

    def load_scenario(self, file_path):
        path = os.path.join(file_path, '*')
        for file in glob.glob(path):
            if not os.path.isdir(file):
                if file.endswith('.json'):
                    scenario_name = os.path.basename(file).split('.')[0]
                    with open(file, 'r', encoding='utf-8') as f:
                        self.scenario = json.load(f)
                        for node in self.scenario:
                            node_id = scenario_name + '-' + node['id']
                            if 'childnode' in node:
                                new_child_id = []
                                for child in node.get('childnode', []):
                                    child_id = scenario_name + '-' + child
                                    new_child_id.append(child_id)
                                node['childnode'] = new_child_id
                            self.node_id_to_node_info[node_id] = node
                    print('场景加载完毕')
            else:
                self.load_scenario(file)

    def slot_template(self, file_path):
        df = pd.read_excel(file_path)
        for index, row in df.iterrows():
            slot = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot] = [query, value]
        print('模板加载完毕')

    def nlu(self, memory):
        # 判断意图是否为空，是否需要保留上一次的hit_intent
        if memory["hit_intent"] != None and "repeat" not in memory["hit_intent"]:
            self.last_intent = memory["hit_intent"]
        # 意图识别
        memory = self.get_intent(memory)
        # 槽位抽取
        if memory["hit_intent"] == self.last_intent:
            memory = self.get_slot(memory)
        return memory

    def get_intent(self, memory):
        # 从所有当前节点中访问节点中得分最高的节点
        max_score = -1
        hit_intent = None
        for node_id in memory["available_nodes"]:
            node = self.node_id_to_node_info[node_id]
            score = self.get_score(node, memory)
            if score > max_score:
                max_score = score
                hit_intent = node_id
        memory["hit_intent"] = hit_intent
        memory["hit_intent_score"] = max_score
        return memory

    def get_score(self, node, memory):
        # 计算当前节点得分
        intent = memory["user_input"]
        node_intents = node["intent"]
        scores = []
        for node_intent in node_intents:
            sentence_similarity = self.get_similarity(node_intent, intent)
            scores.append(sentence_similarity)
        return max(scores)

    def get_similarity(self, node_intent, intent):
        # 计算句子相似度
        set1 = set(node_intent)
        set2 = set(intent)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def get_slot(self, memory):
        # 槽位抽取
        hit_intent = memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            _, values = self.slot_info[slot]
            if re.search(values, memory["user_input"]):
                memory[slot] = re.search(values, memory["user_input"]).group()
        return memory

    def dst(self, memory):
        # 如果用户要重复问题，则返回上一次的答案
        if "repeat" in memory["hit_intent"] and self.last_intent is not None:
            return memory
        # 对话状态跟踪，检测当前intent是否已经被填满
        hit_intent = memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            if slot not in memory:
                memory["need_slot"] = slot
                return memory
        memory["need_slot"] = None
        return memory

    def policy(self, memory):
        # 对话策略，根据当前状态选择下一步动作
        # 如果槽位有欠缺，反问槽位
        # 如果没有欠缺，直接回答
        if "repeat" in memory["hit_intent"] and self.last_intent is not None:
            if "repeat" in memory["hit_intent"]:
                return memory
        if memory["need_slot"] is None:
            memory["action"] = "answer"
            # 如果用户不要求重复回答则开放子节点
            memory["available_nodes"] = self.node_id_to_node_info[memory["hit_intent"]].get("childnode", [])
            memory["available_nodes"].append("scenario-repeat-node1")
        else:
            memory["action"] = "ask"
            # 停留在当前节点
            memory["available_nodes"] = [memory["hit_intent"], "scenario-repeat-node1"]
        return memory

    def replace_slot(self, answer, memory):
        # 填槽
        hit_intent = memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            answer = answer.replace(slot, memory[slot])
        return answer

    def nlg(self, memory):
        # 生成回复
        if "repeat" in memory["hit_intent"] and self.last_intent is not None:
            memory["bot_response"] = memory["bot_response"]
            # print('*'*10, memory)
            return memory
        if memory["action"] == "answer":
            # 回答
            answer = self.node_id_to_node_info[memory["hit_intent"]]["response"]
            # 填槽
            memory["bot_response"] = self.replace_slot(answer, memory)
        else:
            # 反问
            slot = memory["need_slot"]
            query, _ = self.slot_info[slot]
            memory["bot_response"] = query
        # print('-'*10, memory)
        return memory

    def analysis_restart(self, user_input):
        scores = []
        with open(os.path.join(r'E:\badouai\ai\第十六周 对话系统\week16 对话系统\scenario\123\scenario-restart.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for context in data[0]["intent"]:
                score = self.get_similarity(user_input, context)
                scores.append(score)
        if max(scores) > 0.2:
            return True
        return False

    def response_gengerate(self, user_input, memory):
        if self.analysis_restart(user_input):
            memory = copy.deepcopy(self.first_memory)
            _user_input = input("bot: 请输入您的需求:")
            user_input = copy.deepcopy(_user_input)
        memory["user_input"] = user_input
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.policy(memory)
        memory = self.nlg(memory)
        return memory["bot_response"], memory

def create_memory(memory_file_path):
    memory = {"available_nodes":[]}
    get_memory(memory_file_path, memory)
    with open(os.path.join(memory_file_path, '123\\scenario-repeat.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        # memory["available_nodes"].extend(data[0]["repeat"])
    memory["hit_intent"] = None
    return memory

def get_memory(memory_file_path, memory):
    _file_path = os.path.join(memory_file_path, '*')
    for file in glob.glob(_file_path):
        if os.path.isdir(file):
            get_memory(file,memory)
        else:
            if file.endswith('.json'):
                memory["available_nodes"].append(os.path.basename(file).split('.')[0] + '-node1')

if __name__ == '__main__':
    _memory = create_memory(r'E:\badouai\ai\第十六周 对话系统\week16 对话系统\scenario')
    ds = DialogSystem(_memory)
    memory = copy.deepcopy(_memory)
    while True:
        user_input = input("uesr:")
        response, memory = ds.response_gengerate(user_input, memory)
        print()
        print("bot:", response)
        print()