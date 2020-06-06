import pandas as pd
from layers.BiLSTM_Attention import Attention, build_model
from sklearn.model_selection import train_test_split
from utils import Preprocess,load_pre_embedding,RocAucEvaluation
from sklearn.metrics import roc_auc_score

maxlen,embed_dim = 120, 300
preprocessor = Preprocess(maxlen,embed_dim,5)
X_train, y_train, df_test = preprocessor()
X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1025)
embedding_matrix = load_pre_embedding(preprocessor)
model = build_model(preprocessor,embedding_matrix)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

batch_size = 128
epochs=2
model.fit(X_train,y_train,batch_size=batch_size, epochs=epochs, validation_data=(X_val,y_val),callbacks=[RocAuc],verbose=1)

t = model.predict(df_test, batch_size=batch_size,verbose=1)
sub = pd.read_csv('./data/sample_submission.csv.zip',compression='zip')
subm = pd.DataFrame(t,columns=sub.columns[1:])
subm['id'] = sub['id']

subm.to_csv('./output/mySubmission.csv',index=False)
