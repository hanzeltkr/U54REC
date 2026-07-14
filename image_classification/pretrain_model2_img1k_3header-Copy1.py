#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import pickle, random, os
from pathlib import Path


# In[2]:


# Get the images and labels
cpool_imgs = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/cpool_imgs.npy')
cpool_labels = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/cpool_labels.npy')
cpool_meta = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/cpool_metadata.npy')
cpool_header_num = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/cpool_h_num.npy')
cpool_header_str = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/cpool_h_str.npy')

ncpool_imgs = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/ncpool_imgs.npy')
ncpool_labels = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/ncpool_labels.npy')
ncpool_meta = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/ncpool_metadata.npy')
ncpool_header_num = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/ncpool_h_num.npy')
ncpool_header_str = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/ncpool_h_str.npy')

test_imgs = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/test_imgs.npy')
test_labels = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/test_labels.npy')
test_meta = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/test_metadata.npy')
test_header_num = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/test_h_num.npy')
test_header_str = np.load('/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/split/test_h_str.npy')


# In[5]:


# Setup data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.2),
])


# In[6]:


# For numeric features
def get_normalization_layer(index, numeric):
    normalizer = layers.Normalization(axis=None)

    # one numeric column by index, keep shape (batch, 1)
    feature = np.asarray(numeric[:, index], dtype=np.float32).reshape(-1, 1)

    normalizer.adapt(feature)
    return normalizer

# For string categoray features
def get_category_encoding_layer(index, string, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    lookup = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    lookup = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature = np.asarray(string[:, index], dtype=str).reshape(-1, 1)

  # Learn the set of possible values and assign them a fixed integer index.
  lookup.adapt(feature)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(lookup(feature))


# In[7]:


all_inputs = []
encoded_features = []

input_col = tf.keras.Input(shape=(1,), dtype=tf.float32)  
layer = get_normalization_layer(0, cpool_header_num)
all_inputs.append(input_col)
encoded_features.append(layer(input_col))

for i in range(cpool_header_str.shape[1]):
    input_col = tf.keras.Input(shape=(1,), dtype=tf.string) 
    layer = get_category_encoding_layer(i, cpool_header_str, dtype="string")
    all_inputs.append(input_col)
    encoded_features.append(layer(input_col))


# In[8]:


# Load in the images
def load_image(path, meta, label, header_num, header_str):
    def read_npz(p, m):
        p = p.decode() # From byte to string
        img = np.load(p)['data']

        window_min = m[0]
        window_max = m[1]
        img = np.clip(img, window_min, window_max)
        img = (img - window_min) / (window_max - window_min) * 255.0
        return img.astype(np.float32)
    
    img = tf.numpy_function(read_npz, [path, meta], tf.float32)
    img = tf.expand_dims(img, axis=-1)
    img.set_shape([1024, 1024, 1])
    num_features = [tf.expand_dims(header_num[i], axis=-1) for i in range(1)]
    str_features = [tf.expand_dims(header_str[i], axis=-1) for i in range(2)]
    
    return tuple([img] + num_features + str_features), label

# Create the dataset
# Take the rest of cancer pool
c_ds = tf.data.Dataset.from_tensor_slices((cpool_imgs, cpool_meta, cpool_labels, cpool_header_num, cpool_header_str))
c_ds = c_ds.shuffle(1000)
n_images = int(len(cpool_imgs) * 0.8235) # To get fina 70% training and 15% val
ctrain_ds = c_ds.take(n_images).cache()
cval_ds = c_ds.skip(n_images).cache()

# Take unique set of images from non-cancer pool
nc_ds = tf.data.Dataset.from_tensor_slices((ncpool_imgs, ncpool_meta, ncpool_labels, ncpool_header_num, ncpool_header_str))
nc_ds = nc_ds.shuffle(len(ncpool_imgs)).take(len(cpool_imgs))
nctrain_ds = nc_ds.take(n_images).cache()
ncval_ds = nc_ds.skip(n_images).cache()


# In[9]:


# Check balance before map
raw_train = ctrain_ds.concatenate(nctrain_ds)
train_labels_list = [label.numpy() for _, _, label, _, _ in raw_train]
train_labels_arr = np.array(train_labels_list).astype(int)

neg, pos = np.bincount(train_labels_arr)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# In[10]:


# Merge and shuffle
train_ds = ctrain_ds.concatenate(nctrain_ds)
train_size = n_images * 2
train_ds = train_ds.shuffle(train_size)
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32)
# Applies augmentation on the training dataset
train_ds = train_ds.map(
    lambda inputs, y: (
        (data_augmentation(inputs[0], training=True),) + tuple(inputs[1:]), y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = cval_ds.concatenate(ncval_ds)
val_ds = val_ds.shuffle(train_size)
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_meta, test_labels, test_header_num, test_header_str))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32)


# In[11]:


# Configure dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# In[12]:


# Import pretrained model
base_model = tf.keras.applications.ResNet50(input_shape=(1024, 1024, 3),
                                                  include_top=False, # Don't include ImageNet classifier at the top,
                                                  weights='imagenet',
                                                 )
# Freeze base model
base_model.trainable = False


# In[13]:


base_model.summary()


# In[14]:


# Image branch
# Convert image tensor from 1 channel to 3 for the model
image_input = tf.keras.Input(shape=(1024, 1024, 1), name = 'image_input')
#x = data_augmentation(image_input)
x = tf.keras.layers.Concatenate()([image_input, image_input, image_input])
x = preprocess_input(x) 

# Keep base model in inference mode
x = base_model(x, training = False)
x = layers.GlobalAveragePooling2D()(x) # (batch, H, W, C) becomes (batch, C) — a single feature vector per image
x = layers.Dense(64, activation="relu",
                kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.2)(x)

# Headers branch
header_features = layers.concatenate(encoded_features)
h = layers.Dense(32, activation="relu",
                kernel_regularizer=regularizers.l2(1e-4))(header_features)
h = layers.Dropout(0.2)(h)

combined = layers.concatenate([x, h])
outputs = layers.Dense(1, activation="sigmoid")(combined)   # add activation="sigmoid" # maps feature vectors to probability score 0-1
model = tf.keras.Model(inputs=[image_input] + all_inputs, outputs=outputs) # wrap all layers


# In[15]:


# Show input pipeline
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)


# In[17]:


model.summary()


# In[29]:


# Set up learning rate decay with linear warmup
steps_per_epoch = 322
epochs_phase = 20
total_steps = steps_per_epoch * epochs_phase

warmup_steps = int(0.1 * total_steps) 
base_lr = 3e-3                         
finetune_lr = 3e-5

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,
    decay_steps=total_steps - warmup_steps,
    warmup_target=base_lr,
    warmup_steps=warmup_steps,
    alpha=0.1
)


# In[31]:


# Show the learning rate decay
steps = np.arange(total_steps, dtype=np.int32)
lr = np.array([float(lr_schedule(int(s)).numpy()) for s in steps], dtype=np.float32)
epochs_x = steps / steps_per_epoch

plt.figure(figsize=(8, 6))
plt.plot(epochs_x, lr)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(alpha=0.3)
plt.show()


# In[35]:


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
    ]
)


# In[36]:


# Early stopping callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
]


# In[37]:


initial_epochs = 20
history = model.fit(train_ds, 
                    epochs=initial_epochs, 
                    validation_data = val_ds, 
                    callbacks = callbacks)


# In[ ]:


# See how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))


# In[ ]:


base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - 50

# Unfreeze only the head layers
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Freeze BatchNorm layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# In[ ]:


model.summary()


# In[ ]:


finetune_epochs_phase = 40
total_steps = steps_per_epoch * finetune_epochs_phase
warmup_steps = int(0.1 * total_steps)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,
    decay_steps=total_steps - warmup_steps,
    warmup_target=finetune_lr,
    warmup_steps=warmup_steps,
    alpha=0.1
)


# In[ ]:


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
    ],
)


# In[ ]:


# Train the model
fine_tune_epochs = 40
initial_epoch = len(history.history['auc'])
total_epochs = initial_epoch + fine_tune_epochs
history_fine = model.fit(train_ds, 
                         epochs=total_epochs,       
                         initial_epoch=initial_epoch,  
                         validation_data=val_ds,
                         callbacks=callbacks)


# In[ ]:


# Evaluate model
model.evaluate(test_ds)


# In[ ]:


from sklearn.metrics import confusion_matrix

# Analyse confusion matrix
y_prob = model.predict(test_ds).reshape(-1)
y_pred = (y_prob >= 0.5).astype(int)
y_true = test_labels.astype(int)

cm = confusion_matrix(y_true, y_pred)


# In[ ]:


import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# In[ ]:


from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")


# In[ ]:


# Combine both for plotting
acc = history.history['auc'] + history_fine.history['auc']
val_acc = history.history['val_auc'] + history_fine.history['val_auc']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Plot
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training AUC')
plt.plot(val_acc, label='Validation AUC')
plt.ylim([0, 1.0])
plt.plot([initial_epochs,initial_epoch],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation AUC')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs,initial_epoch],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:





# In[ ]:




