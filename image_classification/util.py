#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle, random, os, json
from pathlib import Path
from collections import Counter
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix


# In[ ]:


class Util:
    def __init__(self):
        self.cancer = []
        self.noncancer = []
        self.model = None
        self.base_model = None
        self.train_ds = []
        self.train_headers_num_np = []
        self.train_headers_str_np = []
        self.val_ds = []
        self.test_ds = []
        self.test_labels = []
        self.fine_tune_at = 0

    #Util.load('path/') for train val data
    def load(self, img_path, header_path, random_seed=10):
        # Check image path
        omama_dir_path = Path(img_path) / "images"
        if not os.path.exists(omama_dir_path):
            print(f"File not found: {omama_dir_path}")
            print("Please update dicom_path with a valid image file path")
            return
        else:
            print(f"Found DICOM folder: {os.path.basename(omama_dir_path)}")
            omama_folder = Path(omama_dir_path)
            IDs = sorted(str(file) for file in omama_folder.rglob("*.npz"))

        # Check metadata path
        meta_dir = Path(img_path) / "metadata"

        # Get the headers
        pkl = Path(header_path)
        mapping = pickle.load(open(pkl, "rb"))

        # Get the images, labels, and headers
        noncancer = {}
        cancer = {}
        for ID in IDs :
            meta = json.load(open(meta_dir / f"{Path(ID).stem}.json"))
            if meta["label"] == "Unknown" :
                continue
        
            # Load image
            img_name = ID
            # Load window values
            wc = meta["WindowCenter"]
            ww = meta["WindowWidth"]
            window_center = float(wc[0] if hasattr(wc, '__len__') else wc)
            window_width = float(ww[0] if hasattr(ww, '__len__') else ww)
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2

            # Load headers
            ds = mapping[Path(ID).stem]
            age_str = ds.PatientAge
            if age_str and age_str != '':
                age = float(age_str[:-1]) / 12.0
            else:
                age = -1.0 
            breastimplant = ds.get("BreastImplantPresent")
            if breastimplant == "None" :
                breastimplant = "NO"
            view = ds.ViewPosition
            if meta["label"] == "NonCancer" :
                if meta["PatientID"] not in noncancer.keys() :
                    noncancer[meta["PatientID"]] = []
                noncancer[meta["PatientID"]].append([img_name, [window_min, window_max], 0, 
                                                     [age, breastimplant, view]])
            else : 
                if meta["PatientID"] not in cancer.keys() :
                    cancer[meta["PatientID"]] = []
                cancer[meta["PatientID"]].append([img_name, [window_min, window_max], 1, 
                                                  [age, breastimplant, view]])
        self.noncancer = list(noncancer.items())
        self.cancer = list(cancer.items())

        # Get the random sample of fix seed
        self.random_seed = random_seed
        split_rng = random.Random(self.random_seed)
        split_rng.shuffle(self.cancer)
        split_rng.shuffle(self.noncancer) 

    #Util.load_test('path/') for test data
    def load_test(self, use_meta=True) :  
        # Select cancer patients for test data
        test_amount = int(0.25 * len(self.cancer))
        test_c = self.cancer[:test_amount]
        test_c_data = self._assign_data(test_c)
        
        # Get same numbers of images for non-cancer cases
        test_nc = self.noncancer[:test_amount]
        test_nc_data = self._assign_data(test_nc)
        
        # Merge cancer and non-cancer, shuffle
        test_all = test_c_data + test_nc_data
        random.Random(self.random_seed).shuffle(test_all)

        test_imgs, test_metadata, test_labels, test_header_num, test_header_str = self._np_setup(test_all)
        self.test_labels = test_labels
        if use_meta :
            test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_metadata, test_labels, test_header_num, test_header_str))
        else :
            test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_metadata, test_labels))
        test_ds = test_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.test_ds = test_ds.batch(32)
        
    #Util.sample_data to randomly sample train/val data from large collection
    def sample_data(self, use_meta=True) :
        # Select cancer patients for train/val data
        test_amount = int(0.25 * len(self.cancer))
        trainval_c = self.cancer[test_amount:]
        trainval_c_data = self._assign_data(trainval_c)
        # Get same numbers of images for non-cancer cases
        trainval_nc = self.noncancer[test_amount:]
        trainval_nc_data = self._assign_data(trainval_nc)

        cpool_imgs, cpool_metadata, cpool_labels, cpool_header_num, cpool_header_str = self._np_setup(trainval_c_data)
        ncpool_imgs, ncpool_metadata, ncpool_labels, ncpool_header_num, ncpool_header_str = self._np_setup(trainval_nc_data)

        # Create the dataset
        if use_meta :
            c_ds = tf.data.Dataset.from_tensor_slices((cpool_imgs, cpool_metadata, cpool_labels, cpool_header_num, cpool_header_str))
            nc_ds = tf.data.Dataset.from_tensor_slices((ncpool_imgs, ncpool_metadata, ncpool_labels, ncpool_header_num, ncpool_header_str))
        else :
            c_ds = tf.data.Dataset.from_tensor_slices((cpool_imgs, cpool_metadata, cpool_labels))
            nc_ds = tf.data.Dataset.from_tensor_slices((ncpool_imgs, ncpool_metadata, ncpool_labels))
        # Take the rest of cancer pool
        c_ds = c_ds.shuffle(1000)
        n_images = int((2/3) * len(cpool_imgs))
        ctrain_ds = c_ds.take(n_images).cache()
        cval_ds = c_ds.skip(n_images).cache()
        # Take unique set of images from non-cancer pool
        nc_ds = nc_ds.shuffle(len(ncpool_imgs)).take(len(cpool_imgs))
        nctrain_ds = nc_ds.take(n_images).cache()
        ncval_ds = nc_ds.skip(n_images).cache()

        # Check balance before map
        raw_train = ctrain_ds.concatenate(nctrain_ds)
        if use_meta :
            train_labels_list = [l.numpy() for _, _, l, _, _ in raw_train]
        else :
            train_labels_list = [l.numpy() for _, _, l in raw_train]
        train_labels_arr = np.array(train_labels_list).astype(int)
        neg, pos = np.bincount(train_labels_arr)
        total = neg + pos
        print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))

        # Get the headers from the train dataset for normalizing and encoding later
        if use_meta :
            train_headers_num = [h.numpy() for _, _, _, h, _ in raw_train]
            train_headers_str = [h.numpy() for _, _, _, _, h in raw_train]
            self.train_headers_num_np = np.array(train_headers_num)
            self.train_headers_str_np = np.array(train_headers_str)
        
        # Merge and shuffle
        train_ds = ctrain_ds.concatenate(nctrain_ds)
        val_ds = cval_ds.concatenate(ncval_ds)
        train_size = n_images * 2
        train_ds = train_ds.shuffle(train_size)
        train_ds = train_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.train_ds = train_ds.batch(32)
        val_ds = val_ds.shuffle(train_size)
        val_ds = val_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(32)

    #Util.setupResnet()
    def setupResnet(self, use_meta=True) :
        # Configure dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)

        # Import pretrained model
        self.base_model = tf.keras.applications.ResNet50(input_shape=(1024, 1024, 3),
                                                  include_top=False, # Don't include ImageNet classifier at the top,
                                                  weights='imagenet',
                                                 )
        # Freeze base model
        self.base_model.trainable = False
        self.base_model.summary()
        self.fine_tune_at = len(self.base_model.layers) - 50
        self.model = self._build_model(0.1, use_meta, self.train_headers_num_np, self.train_headers_str_np, resnet_preprocess)
        
        # Show input pipeline and summary
        tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ]
        )


    #Util.setupEfficientNet()
    def setupEfficientNet(self, use_meta=True) :
        # Configure dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)
    
        # Import pretrained model
        self.base_model = tf.keras.applications.EfficientNetB0(input_shape=(1024, 1024, 3),
                                                               include_top=False, # Don't include ImageNet classifier at the top,
                                                               weights='imagenet',
        )
        # Freeze base model
        self.base_model.trainable = False
        self.base_model.summary()
        self.fine_tune_at = len(self.base_model.layers) - 20
        self.model = self._build_model(0.2, use_meta, self.train_headers_num_np, self.train_headers_str_np, lambda x : x)
            
        # Show input pipeline and summary
        tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ]
        )

    def train(self) :
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
        
        initial_epochs = 30
        history = self.model.fit(self.train_ds, 
                  epochs=initial_epochs, 
                  validation_data = self.val_ds, 
                  callbacks = callbacks)

        self.base_model.trainable = True
        # Unfreeze only the head layers
        for layer in self.base_model.layers[:self.fine_tune_at]:
            layer.trainable = False
        # Freeze BatchNorm layers
        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self.model.summary()

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
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=3)
        ]

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5, weight_decay = 0.01),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ],
        )

        # Train the model
        fine_tune_epochs = 40
        initial_epoch = len(history.history['auc'])
        total_epochs = initial_epoch + fine_tune_epochs
        self.model.fit(self.train_ds, 
                  epochs=total_epochs,       
                  initial_epoch=initial_epoch,  
                  validation_data=self.val_ds,
                  callbacks=callbacks)

    def predict(self) :
        best_threshold = self._best_threshold()
        self.model.evaluate(self.test_ds)
        
        # Analyse confusion matrix
        y_prob = self.model.predict(self.test_ds).reshape(-1)
        y_pred = (y_prob >= best_threshold).astype(int)
        y_true = self.test_labels.astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1:.4f}")
    
    # Assign the images, metadata, and labels for setting up mix input
    def _assign_data(self, dataset) :
        all_patient = []
        for patient, data_list in dataset :
            for data in data_list :
                img = data[0]
                metadata = data[1]
                label = data[2]
                headers = data[3]
                all_patient.append((img, metadata, label, headers))
        return all_patient

    # Set up numpy array
    def _np_setup(self, dataset) :
        imgs, metadata, labels, headers_num, headers_str = [],[],[],[],[]
        for data in dataset :
            imgs.append(data[0])
            metadata.append(data[1])
            labels.append(data[2])
            headers_num.append([data[3][0]])
            headers_str.append(list(data[3][1:]))
        imgs_np = np.array(imgs)
        metadata_np = np.array(metadata, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        headers_num_np = np.array(headers_num, dtype=np.float32)
        headers_str_np = np.array(headers_str, dtype=str)
        return imgs_np, metadata_np, labels_np, headers_num_np, headers_str_np

    # For numeric features
    def _get_normalization_layer(self, index, numeric):
        normalizer = layers.Normalization(axis=None)

        # one numeric column by index, keep shape (batch, 1)
        feature = np.asarray(numeric[:, index], dtype=np.float32).reshape(-1, 1)

        normalizer.adapt(feature)
        return normalizer

    # For string categoray features
    def _get_category_encoding_layer(self, index, string, dtype, max_tokens=None):
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
        # layer to use or include them in the Keras Functional model later.
        return lambda feature: encoder(lookup(feature))

    # Load in the images
    def _load_image(self, path, meta, label, header_num=None, header_str=None, use_meta=True):
        def read_npz(p, m):
            p = p.decode() # From byte to string
            img = np.load(p)['data']

            window_min = m[0]
            window_max = m[1]
            img = np.clip(img, window_min, window_max)
            img = (img.astype(np.float32) - window_min) / (window_max - window_min) * 255
            return img.astype(np.float32)
    
        img = tf.numpy_function(read_npz, [path, meta], tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img.set_shape([1024, 1024, 1])

        if use_meta :
            num_features = [tf.expand_dims(header_num[i], axis=-1) for i in range(1)]
            str_features = [tf.expand_dims(header_str[i], axis=-1) for i in range(2)]
            return tuple([img] + num_features + str_features), label
        return (img,), label

    def _load_image_fn(self, use_meta):
        def wrapper(*args):
            return self._load_image(*args, use_meta=use_meta)
        return wrapper

    def _build_model(self, dropout_rate, use_meta, header_num, header_str, preprocess_fn) : 
        all_inputs = []
        encoded_features = []
        if use_meta : 
            input_col = tf.keras.Input(shape=(1,), dtype=tf.float32)
            layer = self._get_normalization_layer(0, header_num)
            all_inputs.append(input_col)
            encoded_features.append(layer(input_col))

            for i in range(header_str.shape[1]):
                input_col = tf.keras.Input(shape=(1,), dtype=tf.string) 
                layer = self._get_category_encoding_layer(i, header_str, dtype="string")
                all_inputs.append(input_col)
                encoded_features.append(layer(input_col))
        
        # Setup data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        # Image branch
        # Convert image tensor from 1 channel to 3 for the model
        image_input = tf.keras.Input(shape=(1024, 1024, 1), name = 'image_input')
        x = data_augmentation(image_input)
        x = tf.keras.layers.Concatenate()([x, x, x])
        x = preprocess_fn(x)

        # Keep base model in inference mode
        x = self.base_model(x, training = False)
        x = layers.GlobalAveragePooling2D()(x) # (batch, H, W, C) becomes (batch, C) — a single feature vector per image
        #x = layers.Dense(64, activation="relu",
        #                kernel_regularizer=regularizers.l2(1e-4))(x)
        combined = layers.Dropout(dropout_rate)(x)

        if use_meta :
            # Headers branch
            header_features = layers.concatenate(encoded_features)
            h = layers.Dense(32, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(header_features)
            h = layers.Dropout(dropout_rate)(h)

            combined = layers.concatenate([x, h])

        # add activation="sigmoid" # maps feature vectors to probability score 0-1
        outputs = layers.Dense(1, activation="sigmoid")(combined) 
        model = tf.keras.Model(inputs=[image_input] + all_inputs, outputs=outputs) # wrap all layers
        return model

    def _best_threshold(self) :
        y_val_probs = self.model.predict(self.val_ds).reshape(-1)
        y_val_true = np.concatenate([y.numpy().reshape(-1) for _, y in self.val_ds]).astype(int)

        # Try thresholds from 0.01 to 0.99
        best_threshold = 0.5
        best_f1 = 0.0

        # Test different thresholds
        for threshold in np.arange(0.1, 1.0, 0.1):
            # Convert probabilities to hard 1s and 0s based on the current threshold
            y_pred_thresholded = (y_val_probs >= threshold).astype(int)
    
            # Calculate F1 score for this threshold
            current_f1 = f1_score(y_val_true, y_pred_thresholded)
    
            # Update the best one
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        print(f"Best Threshold Found: {best_threshold:.2f}")
        print(f"Max Validation F1-Score: {best_f1:.4f}")
        return best_threshold

