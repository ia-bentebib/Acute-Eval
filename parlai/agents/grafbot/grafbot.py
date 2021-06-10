#!/usr/bin/env python3

from parlai.structure.EpiKG import EpiKG
from parlai.structure.SemKG import SemKG
from parlai.tools.Converter import Entities2Tuples
from parlai.tools.EntityExtractor import get_entities
from parlai.core.agents import Agent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.message import Message

class GrafbotAgent(TransformerGeneratorAgent):
    """
    Information Retrieval baseline.
    """
    semkg = SemKG()
    epikg = EpiKG()

    def __init__(self, opt, shared=None):
        """
        Initialize agent.
        """
        super().__init__(opt)
        self.id = 'Grafbot'
        self.opt = opt
        self.learn_file("parlai/agents/grafbot/things.txt")

    def learn_file(self,filepath):
        file = open(filepath, encoding='utf-8')
        things = file.readlines()
        for thing in things:
            thing = thing.replace('\n','').split(';')
            entities = get_entities(thing[0])
            tuples = Entities2Tuples(entities, "linear")
            self.semkg.add_relations(tuples, self.epikg, thing)

    def learn(self,sentences):
            for sentence in sentences:
                entities = get_entities(sentence)
                tuples = Entities2Tuples(entities, "linear")
                self.semkg.add_relations(tuples,self.epikg,sentence)

    def observe(self, obs):
        """
        Store and remember incoming observation message dict.
        """
        split_txt = obs['text'].split('\n')
        persona_txt = []
        answer_txt = []
        new_text = ""
        for text in split_txt:
            if 'your persona: ' in text:
                persona_txt.append(text)
            else:
                answer_txt.append(text)
        if len(persona_txt)>0:
            new_text = '\n'.join(persona_txt) + '\n'
        entities = get_entities('\n'.join(answer_txt))
        stories = self.semkg.get_stories(self.epikg, [x[0] for x in entities])
        new_text += '\n'.join(['your persona: '+text for text in stories if len(text)>0]) + '\n'
        new_text += '\n'.join(answer_txt)
        obs.update({'text': new_text})
        return super().observe(obs)
