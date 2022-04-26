import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import random
import collections
from scipy import stats

import jieba
import jieba.posseg as pseg
import wordcloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ==============导入数据
data = pd.read_csv('tech_bv_info.csv', encoding='utf-8')
print('============data==========')
print(data)
print(data.info())

# ==============数据预处理
# 硬币、收藏、分享
def str_num_trans(x):
    if x=='投币':
        return 0
    if x=='收藏':
        return 0
    if re.search('万',x) is not None:
        return int(float(x.replace('万',''))*10000)
    return int(x)
data['coin'] = data['coin'].map(str_num_trans)
data['collect'] = data['collect'].map(str_num_trans)
data['share'] = data['share'].map(str_num_trans)

# 简介、评论去站内链接、换行
def text_process(x):
    # 去<a>链接
    idx = re.search('<a class((?:.|\n)*)</a>',x)
    if idx is not None:
        idx = idx.span()
        x_a_sub = x[idx[0]:idx[1]]
        a_sub = re.sub('<a class(.*?)>','',x_a_sub.replace('</a>',''))
        x = x.replace(x_a_sub,a_sub)
    # 去<span>
    x = re.sub('<(/*)span>','',x)
    return x
def desc_process(x):
    if pd.isna(x):
        return x
    return text_process(x)
def comment_process(x):
    if pd.isna(x):
        return x
    comment_list = x.split('|$comment-sep$|')
    new_list = []
    for comment in comment_list:
        new_list.append(text_process(comment))
    return '|$comment-sep$|'.join(new_list)
data['describe'] = data['describe'].map(desc_process)
data['comment'] = data['comment'].map(comment_process)

# tag处理，去空格
def tag_process(x):
    tag_list = x.split('|$tag-sep$|')
    new_list = []
    for tag in tag_list:
        new_list.append(tag.replace(' ','').replace('\n',''))
    new_list = list(set(new_list))
    return '|$tag-sep$|'.join(new_list)
data['tag'] = data['tag'].map(tag_process)

# =================数据描述性分析
def count_table(df):
    table = pd.DataFrame()
    for col in df.columns.tolist():
        if df[col].dtype == object:
            continue
        table.loc[col,'max'] = df[col].max()
        table.loc[col,'min'] = df[col].min()
        table.loc[col,'mean'] = round(df[col].mean(),2)
        table.loc[col,'median'] = df[col].median()
    return table
print('============各字段统计量表格==============')
print(count_table(data))

# log变换后，集中在1w-3w
plt.hist(data['play'].map(math.log).to_numpy(),color='#D2B48C')
plt.xlabel('播放量/log')
plt.ylabel('数量')
plt.title('视频播放量分布')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.show()

# 因为是数码科技专区，男性多可以理解
plt.bar(['男','保密','女'],data['author_sex'].value_counts().tolist(), color=['#BDB76B','#F0E68C','#EEE8AA'])
plt.xlabel('性别')
plt.ylabel('数量')
plt.title('up主性别柱状图')
plt.show()

time_videoNum = data['time'].map(lambda x:x[0:10]).value_counts().sort_index(ascending=True)
plt.bar(x=time_videoNum.index.tolist(), height=time_videoNum.tolist(), color='#FA8072')
plt.xlabel('发布时间')
plt.xticks(time_videoNum.index.tolist()[::15],rotation=70)
plt.ylabel('视频数')
plt.title('视频发布时间的分布')
plt.show()

def if_topic(x):
    if '|$tag-topic$|' in x:
        return 1
    return 0
data['event_join'] = data['tag'].map(if_topic)
plt.boxplot([data.loc[data['event_join']==1,'play'].map(math.log).tolist(),
             data.loc[data['event_join']==0,'play'].map(math.log).tolist()],
            labels=['参加','不参加'],
            flierprops={'markerfacecolor':'#CD5C5C'},
            whiskerprops={'color':'#800000'},
            patch_artist=True,
            boxprops={'facecolor':'#DEB887'},
            medianprops={'color':'#800000'})
plt.xlabel('是否参加创作激励活动')
plt.ylabel('播放量/log')
plt.title('参与活动话题与视频播放量')
plt.show()

print('未参加：{}({:.2f})，参加：{}({:.2f})'.format(data[data['event_join']==0].shape[0],data[data['event_join']==0].shape[0]/data.shape[0],
                                    data[data['event_join']==1].shape[0],data[data['event_join']==1].shape[0]/data.shape[0]))
# t检验
print(
    stats.ttest_ind(data.loc[data['event_join']==0,'play'].map(math.log).to_numpy(), data.loc[data['event_join']==1,'play'].map(math.log).to_numpy())
)

plt.boxplot([data.loc[data['duration']<=60,'play'].map(math.log).tolist(),
             data.loc[(data['duration']>60)&(data['duration']<=300),'play'].map(math.log).tolist(),
             data.loc[(data['duration']>300)&(data['duration']<=600),'play'].map(math.log).tolist(),
             data.loc[(data['duration']>600)&(data['duration']<=900),'play'].map(math.log).tolist(),
             data.loc[(data['duration']>900)&(data['duration']<=1800),'play'].map(math.log).tolist(),
             data.loc[(data['duration']>1800)&(data['duration']<=3600),'play'].map(math.log).tolist(),
             data.loc[data['duration']>3600,'play'].map(math.log).tolist()],
            labels=['1分钟以内','1-5分钟','5-10分钟','10-15分钟','15-30分钟','30-60分钟','1小时以上'],
            flierprops={'markerfacecolor':'#CD5C5C'},
            whiskerprops={'color':'#800000'},
            patch_artist=True,
            boxprops={'facecolor':'#DEB887'},
            medianprops={'color':'#800000'})
plt.xlabel('视频时长')
plt.ylabel('播放量/log')
plt.title('视频时长与视频播放量')
plt.show()

plt.scatter(data['author_fans'].map(lambda x:math.log(x) if x!=0 else 0).tolist(), data['play'].map(math.log).tolist(),
            s=4,c='#BDB76B')
plt.xlabel('粉丝数/log')
plt.ylabel('播放量/log')
plt.title('粉丝量与视频播放量')
plt.show()

with open('my_stop_words.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().split('\n')
title_list = data['title'].tolist()
title_words = []
for i in title_list:
    for word in jieba.cut(i):
        if word not in stop_words and len(re.findall('[0-9]',word))!=len(word):
            title_words.append(word)
title_word_freq = sorted(collections.Counter(title_words).items(), key=lambda x:x[1], reverse=True)

title_words_wc = ' '.join(title_words)

wc=wordcloud.WordCloud(font_path="C:\Windows\Fonts\simfang.ttf",width=1000, height=800,
                             max_words=100,
                             background_color='white',colormap='summer')
wc.generate_from_text(title_words_wc)

plt.imshow(wc)

plt.bar(['小米','华为','苹果','iPhone','三星','红米','荣耀'],[269,205,191,147,123,95,70],
        color=['#006400','#008000','#008000','#228B22','#32CD32','#00FF00','#00FF7F'])
plt.xlabel('手机品牌')
plt.ylabel('频数')
plt.title('不同手机品牌出现频数')
plt.show()

phone = []
pc = []
gpu = []
others = []
for i in range(data.shape[0]):
    if '手机' in data.loc[i,'title']:
        phone.append(data.loc[i,'play'])
    elif '显卡' in data.loc[i,'title']:
        gpu.append(data.loc[i,'play'])
    elif '笔记本' in data.loc[i,'title']:
        pc.append(data.loc[i,'play'])
    else:
        others.append(data.loc[i,'play'])
plt.bar(height=[np.array(phone).mean(), np.array(pc).mean(), np.array(gpu).mean(), np.array(others).mean()],
            x=['手机','笔记本','显卡','其他'],color=['#4169E1','#6495ED','#B0C4DE','#708090'])
plt.xlabel('视频类型')
plt.ylabel('平均播放量')
plt.show()
# 标题带有手机、笔记本、显卡的视频平均播放量明显较高

# ========================文本分析
hot_topic = []
hot_bv = []
for i in range(data.shape[0]):
    if data.loc[i,'play']>1000000:
        hot_bv.append(data.loc[i,'bv'])
        hot_video = []
        for word in jieba.cut(data.loc[i,'title']):
            if word not in stop_words:
                hot_video.append(word)
        for word in data.loc[i,'tag'].split('|$tag-sep$|'):
            if word not in hot_video:
                hot_video.append(word)
        hot_topic+=hot_video
hot_topic = sorted(collections.Counter(hot_topic).items(), key=lambda x:x[1], reverse=True)
print(hot_topic)

hot_wc = []
for i in range(11):
    topic=['科技猎手','苹果','华为','安卓','创意','DIY','自制','AMD','性能','发布会','三星']
    topic_num=[83,28,15,15,7,7,7,6,6,6,5]
    for k in range(topic_num[i]):
        hot_wc.append(topic[i])
random.shuffle(hot_wc)
hot_wc = ' '.join(hot_wc)
wc=wordcloud.WordCloud(font_path="C:\Windows\Fonts\simfang.ttf",width=500, height=400,
                             max_words=100,
                             background_color='white',colormap='spring')
wc.generate_from_text(hot_wc)

plt.imshow(wc)

def comment_process2(x):
    if pd.isna(x):
        return x
    com_li = x.split('|$comment-sep$|')
    new_li = []
    for com in com_li:
        new_li.append(com.replace('\n','。').replace('|$comment-top$|',''))
    return '。'.join(new_li)
data['comment_processed'] = data['comment'].map(comment_process2)

com_wc = []
for com in data['comment_processed']:
    if not pd.isna(com):
        for word,flag in pseg.cut(com):
            if word not in stop_words and word not in ['…','_',':'] and word not in ['doge','辣','眼睛','藏狐'] and 'n' in flag:
                com_wc.append(word)
comment_word_freq = sorted(collections.Counter(com_wc).items(), key=lambda x:x[1], reverse=True)

comment_wc = ' '.join(com_wc)

wc=wordcloud.WordCloud(font_path="C:\Windows\Fonts\simfang.ttf",width=1000, height=800,
                             max_words=100,
                             background_color='white',colormap='copper')
wc.generate_from_text(comment_wc)

plt.imshow(wc)

com_wc = []
for com in data['comment_processed']:
    if not pd.isna(com):
        for word in jieba.cut(com):
            if word not in stop_words and word not in ['…','_',':']:
                com_wc.append(word)
comment_word_freq = sorted(collections.Counter(com_wc).items(), key=lambda x:x[1], reverse=True)
wf1 = []
wf2 = []
for i in title_word_freq[:50]:
    wf1.append(i[0])
for i in comment_word_freq[:100]:
    wf2.append(i[0])
wf1s = set(wf1)
wf2s = set(wf2)
wf = wf2s-wf1s
com_unique_freq = []
for i in wf:
    if len(re.findall('[0-9]',i))!=len(i) and len(re.findall('[a-zA-Z]',i))!=len(i) and i not in '说好很没会真的人想笑哭做去更下感觉再里后才很多辣眼睛太中脱单藏狐出东西妙一点不错确实真希望建议只能':
        com_unique_freq.append(comment_word_freq[wf2.index(i)])
com_unique_freq = sorted(com_unique_freq, key=lambda x:x[1], reverse=True)

plt.figure(figsize=(10,6))
plt.bar([i[0] for i in com_unique_freq], [i[1] for i in com_unique_freq],
        color=['#BDB76B','#EEE8AA','#F0E68C','#FFFACD','#FFF8DC'])
plt.xlabel('热评关键词')
plt.ylabel('词频')
plt.title('热评')
plt.show()

# ===================聚类推荐系统
def describe_process2(x):
    if pd.isna(x):
        return x
    return x.replace('\n','。')
data['describe_processed'] = data['describe'].map(describe_process2)
def tag_process2(x):
    if pd.isna(x):
        return x
    return x.replace('|$tag-sep$|','。').replace('|$tag-topic$|','')
data['tag_processed'] = data['tag'].map(tag_process2)

document = []
for i in range(data.shape[0]):
    doc = []
    for word in jieba.cut(data.loc[i,'title']):
        if word not in stop_words and word not in ['…','_',':']:
            doc.append(word)
    if not pd.isna(data.loc[i,'tag_processed']):
        for word in jieba.cut(data.loc[i,'tag_processed']):
            if word not in stop_words and word not in ['…','_',':']:
                doc.append(word)
    if not pd.isna(data.loc[i,'describe_processed']):
        for word in jieba.cut(data.loc[i,'describe_processed']):
            if word not in stop_words and word not in ['…','_',':']:
                doc.append(word)
    if not pd.isna(data.loc[i,'comment_processed']):
        for word in jieba.cut(data.loc[i,'comment_processed']):
            if word not in stop_words and word not in ['…','_',':']:
                doc.append(word)
    document.append(' '.join(doc))

tfidf = TfidfVectorizer()
video_mat = tfidf.fit_transform(document)
svd = TruncatedSVD(n_components=240)
video_mat_svd = svd.fit_transform(video_mat)
km = KMeans(n_clusters=100)
video_clu = km.fit_predict(video_mat_svd)
data['clu']=video_clu
print('==================聚类为第一类的视频====================')
print(data.loc[data['clu']==1,['bv','title','play','like','duration']])