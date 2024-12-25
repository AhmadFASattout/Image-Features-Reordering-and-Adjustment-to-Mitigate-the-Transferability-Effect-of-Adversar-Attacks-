import pickle
from MNIST_Util_filt_clahe import filtering
import numpy as np
import matplotlib.pyplot as plt
from MNIST_64 import MNISTDataset
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import BatchNormalization
import keras
import cv2
import time

class Split_joint_all_detector:

    def __init__(self):
        #Load U_Net Model
        self.dim     = 64
        IMG_WIDTH    = self.dim
        IMG_HEIGHT   = self.dim
        IMG_CHANNELS = 1
        #Build the model
        inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # s = keras.layers.Lambda(lambda x: x / 255)(inputs)

        #Contraction path
        c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        b1 = BatchNormalization()(c1)
        c1 = keras.layers.Dropout(0.1)(b1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        b1 = BatchNormalization()(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(b1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        b2 = BatchNormalization()(c2)
        c2 = keras.layers.Dropout(0.1)(b2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        b2 = BatchNormalization()(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(b2)
        
        c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        b3 = BatchNormalization()(c3)
        c3 = keras.layers.Dropout(0.2)(b3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        b3 = BatchNormalization()(c3)
        p3 = keras.layers.MaxPooling2D((2, 2))(b3)
        
        c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        b4 = BatchNormalization()(c4)
        c4 = keras.layers.Dropout(0.2)(b4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        b4 = BatchNormalization()(c4)
        p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(b4)
        
        c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        b5 = BatchNormalization()(c5)
        c5 = keras.layers.Dropout(0.3)(b5)
        c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        b5 = BatchNormalization()(c5)

        #Expansive path 
        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b5)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        b6 = BatchNormalization()(c6)
        c6 = keras.layers.Dropout(0.2)(b6)
        c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        b6 = BatchNormalization()(c6)
        
        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        b7 = BatchNormalization()(c7)
        c7 = keras.layers.Dropout(0.2)(b7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        b7 = BatchNormalization()(c7)
        
        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(b7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        b8 = BatchNormalization()(c8)
        c8 = keras.layers.Dropout(0.1)(b8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        b8 = BatchNormalization()(c8)
        
        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(b8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        b9 = BatchNormalization()(c9)
        c9 = keras.layers.Dropout(0.1)(b9)
        c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        b9 = BatchNormalization()(c9)
        
        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(b9)

        self.U_Net_Model = keras.models.Model(inputs=[inputs], outputs=[outputs])
        self.U_Net_Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.U_Net_Model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/MNIST_Dataset/checkpointsUnet/allTset_ep_30_acc0.97981_MNIST_UNet.h5")
        self.U_Net_Model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/MNIST_Dataset/checkpointsUnet/MNIST_Unet_first_30_ep_plus_30_all.h5")
        # self.U_Net_Model.load_weights("MNIST_U_Net_with_DC.h5")
        #Load One Class SVM
        # filename = 'C:/Users/ahmad/Desktop/JupyterFilesAUB/MNIST_Dataset/checkpointsSVM/one_svm_model_MNIST_.sav'# Nu of OCSVM is 0.01
        filename = 'C:/Users/ahmad/Desktop/JupyterFilesAUB/MNIST_Dataset/checkpointsSVM/one_svm_model_MNIST_1.sav'# Nu of OCSVM is 0.03 strict more
        # filename   = 'One_SVM_MNIST_model_DC.sav'
        self.OC_SVM_model = pickle.load(open(filename, 'rb'))

        #Load Mask Segmentation Model
        # lrt = 1e-3
        # optimizer = Adam(lr=lrt) # Using Adam instead of SGD to speed up training
        dataset = MNISTDataset()
        self.split_model = dataset.load_model_by_name('carlini', logits=False, input_range_type=1)
        # self.split_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        self.split_model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/MNIST_Dataset/checkpointsSplit_masked_MFilter_noDum_Clahe/MNIST_Split_first_30_ep.h5")
        #Load Segmentation Routine
    def train(self,X_test):
        print("Train is Done")
        print("The Maximum value of this dataset is:",np.array(X_test).max())

    def test(self,X_test,Y_test,adv_cls):
        Split_start_timeP1 = time.clock()
        X_test_ = []
        for img in X_test:
            image = cv2.resize(img, (self.dim, self.dim)) #64*64*1 Images
            img_uint8 = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
            image = np.expand_dims(img_uint8, axis=2)
            X_test_.append(image)
        X_test_ = np.array(X_test_)
        # print("The Maximum value of this image is:",np.array(X_test_).max())
#########################################################
        X_test_U = X_test_/255
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        adv_labels_SVM = Y_test
        X_adv_OCsvm = X_test_masks.reshape((len(X_test_masks), -1))
        result = self.OC_SVM_model.predict(X_adv_OCsvm )
        oc_indices = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
######################################################## 
        X_test_overlay = []
        for i in range(len(X_test_masks)):
           X_test_overlay.append(X_test_[i] * X_test_masks[i] )  
########################################################
        print(X_test_overlay[0].shape)
        height, width, _ = X_test_overlay[0].shape
        img = X_test_overlay[9]
        width_cutoff = width // 2
        s3 = np.zeros_like(X_test_overlay[0])
########################################################
        X_test_swap1 = []
        for img in X_test_overlay:
            # img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap1.append(s3)
        X_test_swap1 = np.array(filtering(X_test_swap1))
        X_test_swap2 = []
        for img in X_test_overlay:
            img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap2.append(s3)
        X_test_swap2   = np.array(filtering(X_test_swap2))
        print(time.clock() - Split_start_time, "Split Detection seconds")
        res = self.detect_adversarial(X_test_swap1, X_test_swap2, adv_labels_SVM, adv_cls)
        return res
    def detect_adversarial(self,X_test_swap1 ,X_test_swap2, adv_labels,adv_cls):
        adv_final_split1_pred  = self.split_model.predict(X_test_swap1/255.,verbose=0)
        adv_final_split2_pred  = self.split_model.predict(X_test_swap2/255., verbose=0)
        adv_final_split1_Cls   = np.argmax(adv_final_split1_pred, axis=1)
        adv_final_split2_Cls   = np.argmax(adv_final_split2_pred, axis=1)

        indeces_eq             = np.where(adv_final_split1_Cls==adv_final_split2_Cls)[0]
        print(len(adv_final_split1_Cls),len(adv_labels),len(indeces_eq))
        pred_split_plus = []
        for i in range(len(adv_labels)):
            if i in indeces_eq:
                # print(adv_labels[i],adv_cls)
                if(adv_labels[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif(adv_final_split1_Cls[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif adv_labels[i] == adv_final_split1_Cls[i]:
                    #successful_Attack , Benign_sample
                    pred_split_plus.append(False)
                else:
                    #Detected_Attack
                    pred_split_plus.append(True)
            else:
                #already two different prediction Detected Attack
                pred_split_plus.append(True)
        # print(len(pred_split_plus))
        pred_split_plus   = np.array(pred_split_plus)
        detected_examples   = len(np.where(pred_split_plus==True)[0])
        successful_examples = len(np.where(pred_split_plus==False)[0])
        print("Detection Rate                                         {:.2f}% ".format(detected_examples/pred_split_plus.shape[0]*100))
        print("Successful Rate                                        {:.2f}% ".format(successful_examples/pred_split_plus.shape[0]*100))
        print("Successful Examples                                   ",successful_examples ,"out of", 
            pred_split_plus.shape[0])
        # return np.where(pred_split_plus==False)[0] , adv_final_seg_Cls, adv_final_inv_Cls,adv_final_seg_pred,adv_final_inv_pred 
        return pred_split_plus
    #########################################################################################
    def test_time(self,X_test,Y_test,adv_cls):
        time_all = 0
        Split_start_timeP1 = time.clock()
        X_test_ = []
        for img in X_test:
            image = cv2.resize(img, (self.dim, self.dim)) #64*64*1 Images
            img_uint8 = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
            image = np.expand_dims(img_uint8, axis=2)
            X_test_.append(image)
        X_test_ = np.array(X_test_)
        # print("The Maximum value of this image is:",np.array(X_test_).max())
#########################################################
        X_test_U = X_test_/255
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        adv_labels_SVM = Y_test
        X_adv_OCsvm = X_test_masks.reshape((len(X_test_masks), -1))
        result = self.OC_SVM_model.predict(X_adv_OCsvm )
        oc_indices = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
######################################################## 
        X_test_overlay = []
        for i in range(len(X_test_masks)):
           X_test_overlay.append(X_test_[i] * X_test_masks[i] )  
########################################################
        print(X_test_overlay[0].shape)
        height, width, _ = X_test_overlay[0].shape
        img = X_test_overlay[9]
        width_cutoff = width // 2
        s3 = np.zeros_like(X_test_overlay[0])
########################################################
        X_test_swap1 = []
        for img in X_test_overlay:
            # img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap1.append(s3)
        X_test_swap1 = np.array(filtering(X_test_swap1))
        X_test_swap2 = []
        for img in X_test_overlay:
            img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap2.append(s3)
        X_test_swap2   = np.array(filtering(X_test_swap2))
        time_all += (time.clock() - Split_start_timeP1)
        print(time_all, "Split Detection seconds p1 time")
        res = self.detect_adversarial_time(X_test_swap1, X_test_swap2, adv_labels_SVM, adv_cls, time_all)
        return res
    def detect_adversarial_time(self,X_test_swap1 ,X_test_swap2, adv_labels,adv_cls, time_all):
        Split_start_timeP2 = time.clock()
        adv_final_split1_pred  = self.split_model.predict(X_test_swap1/255.,verbose=0)
        adv_final_split2_pred  = self.split_model.predict(X_test_swap2/255., verbose=0)
        adv_final_split1_Cls   = np.argmax(adv_final_split1_pred, axis=1)
        adv_final_split2_Cls   = np.argmax(adv_final_split2_pred, axis=1)

        indeces_eq             = np.where(adv_final_split1_Cls==adv_final_split2_Cls)[0]
        print(len(adv_final_split1_Cls),len(adv_labels),len(indeces_eq))
        pred_split_plus = []
        for i in range(len(adv_labels)):
            if i in indeces_eq:
                # print(adv_labels[i],adv_cls)
                if(adv_labels[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif(adv_final_split1_Cls[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif adv_labels[i] == adv_final_split1_Cls[i]:
                    #successful_Attack , Benign_sample
                    pred_split_plus.append(False)
                else:
                    #Detected_Attack
                    pred_split_plus.append(True)
            else:
                #already two different prediction Detected Attack
                pred_split_plus.append(True)
        time_all += (time.clock() - Split_start_timeP2)
        print(time_all, "Split Detection seconds p2 time")
        # print(len(pred_split_plus))
        pred_split_plus   = np.array(pred_split_plus)
        detected_examples   = len(np.where(pred_split_plus==True)[0])
        successful_examples = len(np.where(pred_split_plus==False)[0])
        print("Detection Rate                                         {:.2f}% ".format(detected_examples/pred_split_plus.shape[0]*100))
        print("Successful Rate                                        {:.2f}% ".format(successful_examples/pred_split_plus.shape[0]*100))
        print("Successful Examples                                   ",successful_examples ,"out of", 
            pred_split_plus.shape[0])
        # return np.where(pred_split_plus==False)[0] , adv_final_seg_Cls, adv_final_inv_Cls,adv_final_seg_pred,adv_final_inv_pred 
        return pred_split_plus
    #########################################################################################
    def test_visualize_Verifier(self,X_test,X_test_orig,Y_test,adv_cls,exp_name):
        X_test_      = []
        X_test_orig_ = []
        count        = 0
        for img in X_test:
            image = cv2.resize(img, (self.dim, self.dim)) #64*64*1 Images
            img_uint8 = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
            image = np.expand_dims(img_uint8, axis=2)
            X_test_.append(image)

            image_orig = cv2.resize(X_test_orig[count], (self.dim, self.dim)) #64*64*1 Images
            img_uint8_orig = np.clip(np.rint(image_orig * 255), 0, 255).astype(np.uint8)
            image_orig = np.expand_dims(img_uint8_orig, axis=2)
            X_test_orig_.append(image_orig)
            count+=1

        X_test_       = np.array(X_test_)
        X_test_U      = X_test_/255.
        X_test_orig_  = np.array(X_test_orig_)
        X_test_orig_U = X_test_orig_/255.
        # print("Utest",np.array(X_test_U).max())
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        X_test_overlay_mask  = []
        X_test_mask_orig           = self.U_Net_Model.predict(X_test_orig_U,verbose=0)
        X_test_preds_mask_t_orig   = (X_test_mask_orig > 0.5).astype(np.uint8)
        X_test_masks_orig          = np.array(X_test_preds_mask_t_orig)
        X_test_overlay_mask_orig  = []
        for i in range(len(X_test_masks)):
            X_test_overlay_mask.append(X_test_[i] * X_test_masks[i])
        for i in range(len(X_test_masks_orig)):
            X_test_overlay_mask_orig.append(X_test_orig_[i] * X_test_masks_orig[i])
        # print("test_Over",np.array(X_test_overlay_mask).max(),np.array(X_test_overlay_mask).shape)
        adv_labels_SVM = np.array(Y_test)
        X_adv_OCsvm = X_test_masks.reshape((len(X_test_masks), -1))
        result = self.OC_SVM_model.predict(X_adv_OCsvm )
        oc_indices = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
        height, width, _ = X_test_overlay_mask[0].shape
        img = X_test_overlay_mask[9]
        width_cutoff = width // 2
        s3 = np.zeros_like(X_test_overlay_mask[0])
        ########################################################
        X_test_swap1 = []
        for img in X_test_overlay_mask:
            # img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay_mask[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap1.append(s3)
        X_test_swap1 = np.array(filtering(X_test_swap1))
        X_test_swap2 = []
        for img in X_test_overlay_mask:
            img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay_mask[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap2.append(s3)
        X_test_swap2   = np.array(filtering(X_test_swap2))
        ########################################################
        X_test_swap_orig1 = []
        for img in X_test_overlay_mask_orig:
            # img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay_mask_orig[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap_orig1.append(s3)
        X_test_swap_orig1 = np.array(filtering(X_test_swap_orig1))
        X_test_swap_orig2 = []
        for img in X_test_overlay_mask_orig:
            img = np.expand_dims(cv2.flip(img,1),axis=2) # Mirror
            s3 = np.zeros_like(X_test_overlay_mask_orig[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap_orig2.append(s3)
        X_test_swap_orig2   = np.array(filtering(X_test_swap_orig2))

        adv_final_split1_pred  = self.split_model.predict(X_test_swap1/255.,verbose=0)
        adv_final_split2_pred  = self.split_model.predict(X_test_swap2/255., verbose=0)
        adv_final_split1_Cls   = np.argmax(adv_final_split1_pred, axis=1)
        adv_final_split2_Cls   = np.argmax(adv_final_split2_pred, axis=1)

        indeces_eq             = np.where(adv_final_split1_Cls==adv_final_split2_Cls)[0]
        print(len(adv_final_split1_Cls),len(adv_labels_SVM),len(indeces_eq))
        pred_split_plus = []
        j = -1
        verifier_detection_indx = []
        for i in range(len(adv_labels_SVM)):
            j+=1
            if i in indeces_eq:
                # print(adv_labels[i],adv_cls)
                if(adv_labels_SVM[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif(adv_final_split1_Cls[i]==adv_cls):
                    #Detected_Attack
                    pred_split_plus.append(True)
                elif adv_labels_SVM[i] == adv_final_split1_Cls[i]:
                    #successful_Attack , Benign_sample
                    pred_split_plus.append(False)
                else:
                    #Detected_Attack
                    verifier_detection_indx.append(j)
                    pred_split_plus.append(True)
            else:
                #already two different prediction Detected Attack
                verifier_detection_indx.append(j)
                pred_split_plus.append(True)
        # segmented_imgs, inverted_imgs           = local_K_means_seg_1(np.copy(X_test_overlay_mask),percentage)
        # segmented_imgs_orig, inverted_imgs_orig = local_K_means_seg_1(np.copy(X_test_overlay_mask_orig),percentage)
        X_adv_visual        = []
        X_org_visual        = []
        X_adv_visual_swap   = []
        X_org_visual_swap   = []
        X_adv_visual_mirror = []
        X_org_visual_mirror = []        
        for i in range(len(verifier_detection_indx)):
            X_adv_visual       .append(X_test_U         [verifier_detection_indx[i]])
            X_org_visual       .append(X_test_orig_U    [verifier_detection_indx[i]])
            X_adv_visual_swap  .append(X_test_swap1     [verifier_detection_indx[i]])
            X_org_visual_swap  .append(X_test_swap_orig1[verifier_detection_indx[i]])
            X_adv_visual_mirror.append(X_test_swap2     [verifier_detection_indx[i]])
            X_org_visual_mirror.append(X_test_swap_orig2[verifier_detection_indx[i]])
        X_adv_visual        = np.array(X_adv_visual)
        X_org_visual        = np.array(X_org_visual)
        X_adv_visual_swap   = np.array(X_adv_visual_swap)
        X_org_visual_swap   = np.array(X_org_visual_swap)
        X_adv_visual_mirror = np.array(X_adv_visual_mirror)
        X_org_visual_mirror = np.array(X_org_visual_mirror)

        X_adv_visual        = X_adv_visual       [0:min(10,len(X_adv_visual)+1)]
        X_org_visual        = X_org_visual       [0:min(10,len(X_adv_visual)+1)]
        X_adv_visual_swap   = X_adv_visual_swap  [0:min(10,len(X_adv_visual)+1)]
        X_org_visual_swap   = X_org_visual_swap  [0:min(10,len(X_adv_visual)+1)]
        X_adv_visual_mirror = X_adv_visual_mirror[0:min(10,len(X_adv_visual)+1)]
        X_org_visual_mirror = X_org_visual_mirror[0:min(10,len(X_adv_visual)+1)]

        fig, axs = plt.subplots(nrows = 6, ncols = len(X_adv_visual),figsize=(10, 6))
        axs.flatten()
        fig.subplots_adjust(hspace=0, wspace=0)
        for j in range(len(X_adv_visual)):
            axs[0,j].imshow   (X_org_visual[j]    , cmap="gray")
            axs[0,j].axis     ("off")
            # axs[j,0].set_title(x_l_test[j])
            axs[1,j].imshow   (X_adv_visual[j]    , cmap="gray")
            axs[1,j].axis     ("off")
            axs[2,j].imshow   (X_org_visual_swap[j], cmap="gray")
            axs[2,j].axis     ("off")
            axs[3,j].imshow   (X_adv_visual_swap[j], cmap="gray")
            axs[3,j].axis     ("off")
            axs[4,j].imshow   (X_org_visual_mirror[j], cmap="gray")
            axs[4,j].axis     ("off")
            axs[5,j].imshow   (X_adv_visual_mirror[j], cmap="gray")
            axs[5,j].axis     ("off")
        fig.savefig('C:/Users/ahmad/Desktop/First_Paper/Second_paper/Split_Verifier/MNIST_adv_Verifier_'+exp_name+'.jpg')
    def test_visualize_SVM(self,X_test,X_test_orig,Y_test,adv_cls,exp):
        X_test_      = []
        X_test_orig_ = []
        count = 0
        for img in X_test:
            image = cv2.resize(img, (self.dim, self.dim)) #64*64*1 Images
            img_uint8 = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
            image = np.expand_dims(img_uint8, axis=2)
            X_test_.append(image)

            image_orig = cv2.resize(X_test_orig[count], (self.dim, self.dim)) #64*64*1 Images
            img_uint8_orig = np.clip(np.rint(image_orig * 255), 0, 255).astype(np.uint8)
            image_orig = np.expand_dims(img_uint8_orig, axis=2)
            X_test_orig_.append(image_orig)
            count+=1
        X_test_       = np.array(X_test_)
        X_test_U      = X_test_/255.
        X_test_orig_  = np.array(X_test_orig_)
        X_test_orig_U = X_test_orig_/255.
        # print("Utest",np.array(X_test_U).max())
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        # X_test_overlay_mask  = []
        X_test_mask_orig           = self.U_Net_Model.predict(X_test_orig_U,verbose=0)
        X_test_preds_mask_t_orig   = (X_test_mask_orig > 0.5).astype(np.uint8)
        X_test_masks_orig          = np.array(X_test_preds_mask_t_orig)
        # X_test_overlay_mask_orig  = []
        # print("test_",np.array(X_test_).max())
        # for i in range(len(X_test_masks)):
        #     X_test_overlay_mask.append(X_test_[i] * X_test_masks[i])
        # print("test_Over",np.array(X_test_overlay_mask).max(),np.array(X_test_overlay_mask).shape)
        adv_labels_SVM = Y_test
        X_adv_OCsvm = X_test_masks.reshape((len(X_test_masks), -1))
        result = self.OC_SVM_model.predict(X_adv_OCsvm )
        
        # X_adv_OCsvm_orig  = X_test_masks_orig .reshape((len(X_test_masks_orig ), -1))
        # result_orig  = self.OC_SVM_model.predict(X_adv_OCsvm_orig)
        oc_indices = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
        X_adv_visual      = X_test_U         [oc_indices]
        X_org_visual      = X_test_orig_U    [oc_indices]
        X_adv_visual_mask = X_test_masks     [oc_indices]
        X_org_visual_mask = X_test_masks_orig[oc_indices]
        X_adv_visual      = X_adv_visual     [0:min(10,len(oc_indices)+1)]
        X_org_visual      = X_org_visual     [0:min(10,len(oc_indices)+1)]
        X_adv_visual_mask = X_adv_visual_mask[0:min(10,len(oc_indices)+1)]
        X_org_visual_mask = X_org_visual_mask[0:min(10,len(oc_indices)+1)]
        print("SVM_Detected_AdvImages",len(oc_indices))
        # print("Array_shape",np.array(X_org_visual[0]).shape)
        fig, axs = plt.subplots(nrows = 4, ncols = len(X_adv_visual)+1,figsize=(10, 4))
        axs.flatten()
        fig.subplots_adjust(hspace=0, wspace=0)
        for j in range(len(X_adv_visual)):
            # print("Array_shape",np.array(X_org_visual[j]).shape)
            axs[0,j].imshow   (X_org_visual[j],cmap="gray"      )
            axs[0,j].axis     ("off")
            # axs[j,0].set_title(x_l_test[j])
            axs[1,j].imshow   (X_adv_visual[j],cmap="gray")
            axs[1,j].axis     ("off")
            axs[2,j].imshow   (X_org_visual_mask[j]   ,cmap="gray")
            axs[2,j].axis     ("off")
            axs[3,j].imshow   (X_adv_visual_mask[j],cmap="gray")
            axs[3,j].axis     ("off")
        fig.savefig('C:/Users/ahmad/Desktop/First_Paper/MNIST_adv_SVM_'+exp+'.jpg')
    