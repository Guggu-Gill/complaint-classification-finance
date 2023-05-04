import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matriximport scipy
import pandas as pd
import tensorflow_hub as hub
from keras import backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight

#for binarizing target variable
def binarize_target(x):
    lb = preprocessing.LabelBinarizer()
    lb.fit(['No','Yes'])
    #this returns array
    return lb.transform(x)


#helper function to find !A^B
def not_intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

#loading ONE_HOT_ENCODED DATA
X_train_ohe=scipy.sparse.load_npz("/kaggle/input/guggu-jatt/X_train.npy.npz")
X_test_ohe=scipy.sparse.load_npz("/kaggle/input/guggu-jatt/X_test.npy.npz")

print(X_train_ohe.shape,X_test_ohe.shape)

#Loading 512 dimen Transformer Encoder sentence
X_train_embd=np.load("/kaggle/input/guggu-jatt/X_train_nar.npy")
X_test_embd=np.load("/kaggle/input/guggu-jatt/X_test_nar.npy")

#loading y target varriable
y_train=np.load("/kaggle/input/guggu-jatt/y_train.npy")
y_test=np.load("/kaggle/input/guggu-jatt/y_test.npy").ravel()

print(y_train.shape,y_test.shape)

#Individual classification model for One hot encooded data
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
X, y = load_iris(return_X_y=True)
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X_train_ohe.toarray(), y_train)

#AUC=0.5 obtained

unique, counts = np.unique(y_train, return_counts=True)
print(unique[0],"->",counts[0])
print(unique[1],"->",counts[1])


unique, counts = np.unique(y_test, return_counts=True)
print(unique[0],"->",counts[0])
print(unique[1],"->",counts[1])

#custom loss function for imbalance data set
def custom_binary_loss(y_true, y_pred,w_0,w_1): 
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    term_0 = tf.cast((1 - y_true),tf.float32) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1 
    term_1 = tf.cast(y_true,tf.float32) * K.log(y_pred + K.epsilon()) # Cancels out when target is 0

    return -K.mean(tf.cast((w_0*term_0),tf.float32) + tf.cast((w_1*term_1),tf.float32), axis=1)


#Deep network for 722 dimen vector
def deep(optimizer,loss):
    embedding_input=keras.Input(shape=(722,),name="input")
    deep = tf.keras.layers.Dense(256, activation='relu',name="Dense_Deep_256")(embedding_input)
    deep=tf.keras.layers.Dropout(.2)(deep)
    deep = tf.keras.layers.Dense(128, activation='relu',name="Dense_Deep_128")(deep)
    
    deep = tf.keras.layers.Dense(64, activation='relu',name="Dense_Deep_64")(deep)
    deep=tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dense(32, activation='relu',name="Dense_Deep_32")(deep)
    
    deep = tf.keras.layers.Dense(16, activation='relu',name="Dense_Deep_16")(deep)
    both = tf.keras.layers.concatenate([deep],name="concat_16_16")
    
    output = tf.keras.layers.Dense(1, activation='sigmoid',name='sigmoid')(both)
    keras_model = tf.keras.models.Model([embedding_input], output)
    keras_model.compile(optimizer=optimizer,
                    loss=loss,metrics=[loss,tf.keras.metrics.TruePositives()
                    ])
    return keras_model
#wide network
def wide(optimizer,loss):
    one_hot_embd=keras.Input(shape=(722,),name="input",sparse=True)
    wide = tf.keras.layers.Dense(16, activation='relu',name="Wide_16")(one_hot_embd)
    both = tf.keras.layers.concatenate([wide],name="concat_16_16")
    
    output = tf.keras.layers.Dense(1, activation='sigmoid',name='sigmoid')(both)
    keras_model = tf.keras.models.Model([one_hot_embd], output)
    keras_model.compile(optimizer=optimizer,
                    loss=loss,metrics=[loss,tf.keras.metrics.TruePositives()
                    ])
    return keras_model





#Deep & Wide Network for feature enginnered One_Hot_Encoded data & dense sentence embedded data,
#It mimics both generalisation  & Memorisation improving the offline & Online metric
def Deep_wide(optimizer,loss):
    one_hot_embd=tf.keras.Input(shape=(210,),name="one_hot_embd_concated",sparse=True)
    embedding_input=tf.keras.Input(shape=(512,),name="text_embedings")
    deep = tf.keras.layers.Dense(256, activation='relu',name="Dense_Deep_256")(embedding_input)
    deep=tf.keras.layers.Dropout(.2)(deep)
    deep = tf.keras.layers.Dense(128, activation='relu',name="Dense_Deep_128")(deep)
    
    deep = tf.keras.layers.Dense(64, activation='relu',name="Dense_Deep_64")(deep)
    deep=tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dense(32, activation='relu',name="Dense_Deep_32")(deep)
    
    deep = tf.keras.layers.Dense(16, activation='relu',name="Dense_Deep_16")(deep)
    wide = tf.keras.layers.Dense(16, activation='relu',name="Wide_16")(one_hot_embd)
    both = tf.keras.layers.concatenate([deep, wide],name="concat_16_16")
    
    output = tf.keras.layers.Dense(1, activation='sigmoid',name='sigmoid')(both)
    keras_model = tf.keras.models.Model([embedding_input,one_hot_embd], output)
    keras_model.compile(optimizer=optimizer,
                    loss=loss,metrics=[loss,tf.keras.metrics.TruePositives()
                    ])
    return keras_model

#AUC in deep & Wide was 0.61 which was better than single deep or single wide classifier


#using deep & wide model
model=Deep_wide(tf.keras.optimizers.legacy.Adam(learning_rate=0.001),"binary_crossentropy")

#printing model summary
model.summary()

#plotting keras plot
tf.keras.utils.plot_model(model, "my_first_model.png",show_shapes=True)

#getting class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train.ravel())
#training model on train data with validation split
history=model.fit(
    xx={"text_embedings": X_train_embd, "one_hot_embd_concated": X_train_ohe.toarray()},
    y=y_train,validation_split = 0.2,
        batch_size=32, epochs=250,class_weight = {0: class_weights[0],1:class_weights[1]}

)

#plotting val & train loss
history.history.keys()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



#code used for evaluating
test_history=model.evaluate(
    x={"text_embedings": X_test_embd, "one_hot_embd_concated": X_test_ohe.toarray()},
    y=y_test,
    steps=100
)
#saving the model
model.save("deep_wide")

#predicting target using model
y_pred=model.predict( x={"text_embedings": X_test_embd, "one_hot_embd_concated": X_test_ohe.toarray()})

preds=np.where(y_pred> 0.50, 1, 0)



#plotting confusion matrix
cm=confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#printing ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    y_pred
)

plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")