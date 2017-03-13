# CNER - Chinese Named Entity Recognition

This is a python (Keras and TensorFlow in particular) implementation of NER tagging, specializing in Chinese. The model structure is very close to what's described in 
"Named Entity Recognition with Bidirectional LSTM-CNNs" by Chiu, Jason P. C.; Nichols, Eric and "Natural Language Processing (almost) from Scratch" by 
Collobert, Ronan et al. 

The model is trained (for now) on PKU's labeled corpus of People's Daily published in Jan. 1998 ("北大语料库加工规范：切分·词性标注·注音", Journal of Chinese Language and Computing, 13 (2) 121-158). This corpus has a rich set of labels; only "nr", "nrf" and "nrg" for persons, "ns" for locations, and "nt" for organizations are used for this model. 

The performance of the model is slightly worse (for now) than the Stanford NLP's NER module for Chinese. The average FB1 on test data for the Person/Location/Organization categories is around 55% (FB1 (Person) ~ 70%, FB1 (Location) ~56% and FB1 (Organization) ~ 23%). 

## Usage

```python
from ner import *
cner = NER()
texts = [u'钱其琛在讲话中指出，中国和南非建立外交关系，是两国关系史中一个重要里程碑，标志着两国友好合作关系新篇章的开始。“我们同南非朋友一样对两国关系实现正常化感到高兴。我愿借此机会，向长期来为促进中国和南非人民之间的了解和友谊，为积极推动两国关系正常化进程做出不懈努力的各界人士表示诚挚的谢意"。', u'五十年代，杨虎城将军的女儿杨拯陆来到新疆，为勘探石油艰辛备尝，最后献出生命。她的事迹感人至深，她的精神令人落泪。克拉玛依市和新疆石油管理局受杨拯陆的故事启发，最近组织艺术家创作、排演了舞剧《大漠女儿》，歌颂一代石油工业建设者的创业形象。这已成为新疆舞台和石油建设者心目中的一件大事。']
for item in cner.predict(texts):
    print item
```

will generate output
```
<PER>钱其琛</PER>在讲话中指出，<LOC>中国</LOC>和<LOC>南非</LOC>建立外交关系，是两国关系史中一个重要里程碑，标志着两国友好合作关系新篇章的开始。“我们同<LOC>南非</LOC>朋友一样对两国关系实现正常化感到高兴。我愿借此机会，向长期来为促进<LOC>中国</LOC>和<LOC>南非</LOC>人民之间的了解和友谊，为积极推动两国关系正常化进程做出不懈努力的各界人士表示诚挚的谢意"。
五十年代，<PER>杨虎城</PER>将军的女儿<PER>杨拯陆</PER>来到<LOC>新疆</LOC>，为勘探石油艰辛备尝，最后献出生命。她的事迹感人至深，她的精神令人落泪。<PER>克拉玛</PER>依市和新<LOC>疆</LOC>石油管理局受<PER>杨拯陆</PER>的故事启发，最近组织艺术家创作、排演了舞剧《大漠女儿》，歌颂一代石油工业建设者的创业形象。这已成为新<LOC>疆</LOC>舞台和石油建设者心目中的一件大事。

```

