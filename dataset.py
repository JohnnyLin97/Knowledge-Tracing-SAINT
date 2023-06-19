import config 

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split 
import os
import re


class DKTDataset(Dataset):
  def __init__(self,group,n_skills,max_seq = 100):
    self.samples = group
    self.n_skills = n_skills
    self.max_seq = max_seq
    self.data = []

    for que,res_time,correct_rate,ans in self.samples:
        if len(que)>=self.max_seq:
            self.data.extend([(que[l:l+self.max_seq],res_time[l:l+self.max_seq],correct_rate[l:l+self.max_seq],ans[l:l+self.max_seq])\
            for l in range(len(que)) if l%self.max_seq==0])
        elif len(que)<self.max_seq and len(que)>10:
            self.data.append((que,res_time,correct_rate,ans))
        else :
            continue
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    source_quiz,elapsed_time,answerGroup_correct_rate,correct = self.data[idx]
    seq_len = len(source_quiz)

    q_ids = np.zeros(self.max_seq,dtype=int)
    ans = np.zeros(self.max_seq,dtype=int)
    elap_time = np.zeros(self.max_seq,dtype=int)
    ag_correct_rate = np.zeros(self.max_seq,dtype=int)

    if seq_len>=self.max_seq:
      q_ids[:] = source_quiz[-self.max_seq:]
      ans[:] = correct[-self.max_seq:]
      elap_time[:] = elapsed_time[-self.max_seq:]
      ag_correct_rate[:] = answerGroup_correct_rate[-self.max_seq:]
    else:
      q_ids[-seq_len:] = source_quiz
      ans[-seq_len:] = correct
      elap_time[-seq_len:] = elapsed_time
      ag_correct_rate[-seq_len:] = answerGroup_correct_rate
    
    target_qids = q_ids[1:]
    label = ans[1:] 

    input_ids = np.zeros(self.max_seq-1,dtype=int)
    input_ids = q_ids[:-1].copy()

    input_rtime = np.zeros(self.max_seq-1,dtype=int)
    input_rtime = elap_time[:-1].copy()

    input_cat = np.zeros(self.max_seq-1,dtype=int)
    input_cat = ag_correct_rate[:-1].copy()

    input = {"input_ids":input_ids,"input_rtime":input_rtime.astype(np.int),"input_cat":input_cat}

    return input,target_qids,label 



def get_dataloaders():      
    print("loading csv.....")       

    train_df = pd.DataFrame(columns=["timestamp","user_id","source_quiz","source_quizGroup","elapsed_time","answerGroup_correct_rate","correct"])
    directory = os.path.join(config.TRAIN_FOLDER)

    for root,dirs,files in os.walk(directory):
      for filename in files:
        df = pd.read_csv(directory+'/'+filename)
        df = df.drop("tags", axis=1)
        df = df.drop("user_answer", axis=1)
        df = df.drop("source_quizGroup", axis=1)
        user_id = filename.strip('user_')
        user_id = user_id.strip('.csv')
        df['user_id'] = [user_id]*df.shape[0]
        train_df = pd.concat([df, train_df])

    print("shape of dataframe :",train_df.shape) 

    train_df.elapsed_time /= 100000
    train_df.elapsed_time.fillna(0,inplace=True)
    train_df.elapsed_time.clip(lower=0,upper=1,inplace=True)
    
    train_df["source_quiz"] = train_df["source_quiz"].map(lambda x: np.int64(abs(hash(x)) % (10 ** 5))) 

    train_df = train_df.sort_values(["timestamp"],ascending=True).reset_index(drop=True)

    #grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id","source_quiz","elapsed_time","answerGroup_correct_rate","correct"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.source_quiz.values,r.elapsed_time.values,r.answerGroup_correct_rate.values,r.correct.values))
    del train_df
    gc.collect()

    print("splitting")
    train,val = train_test_split(group,test_size=0.2) 
    print("train size: ",train.shape,"validation size: ",val.shape)
    train_dataset = DKTDataset(train.values,n_skills=0,max_seq = config.MAX_SEQ)
    val_dataset = DKTDataset(val.values,n_skills=0,max_seq = config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    del train_dataset,val_dataset
    gc.collect()
    return train_loader, val_loader