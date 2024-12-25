import pickle
from GTSRB_Util_split_MedFilt_Clahe import medfiltering
import numpy as np
from GTSRB_8 import GTSRB8_Dataset
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import BatchNormalization
import keras
import cv2
from skimage import img_as_ubyte
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
        self.U_Net_Model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/GTSRB_dataset/checkpointsUnet/allTset_ep_15_acc0.95195_GTSRB_UNet.h5")
        #Load One Class SVM
        filename = 'C:/Users/ahmad/Desktop/JupyterFilesAUB/GTSRB_dataset/checkpointsSVM/one_svm_model_GTSRB.sav'
        self.OC_SVM_model = pickle.load(open(filename, 'rb'))

        #Load Mask Segmentation Model
        # lrt = 1e-4
        # optimizer = Adam(lr=lrt) # Using Adam instead of SGD to speed up training
        dataset = GTSRB8_Dataset()
        self.split_model = dataset.load_model_by_name('densenet', logits=False, input_range_type=1)
        # self.split_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        self.split_model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/GTSRB_dataset/checkpointsSplit_msk_MedFilter_Nodum_Clahe/ep_41_los0.149_acc0.99601_GTSRB_Split.h5")
        # self.split_model.load_weights("C:/Users/ahmad/Desktop/JupyterFilesAUB/GTSRB_dataset/checkpointsSplit_msk_MedFilter_Nodum_Clahe/GTSRB_Split_first_50_ep.h5")
        #Load Segmentation Routine
    def train(self):
        print("Train is Done")

    def test(self,X_test,Y_test,adv_cls):
        X_test_     = []
        X_test_orig = []
        for img in X_test:
            img_uint8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
            image     = cv2.cvtColor(img_uint8,cv2.COLOR_RGB2GRAY)#Image now is 255-level
            image     = np.expand_dims(image, axis=2)
            X_test_orig.append(img_uint8)
            X_test_.append(image)
        X_test_ = np.array(X_test_)
#########################################################
        X_test_U = X_test_/255.
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        adv_labels_SVM        = np.array(Y_test)
        X_adv_OCsvm           = X_test_masks.reshape((len(X_test_masks), -1))
        result                = self.OC_SVM_model.predict(X_adv_OCsvm )
        oc_indices            = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
######################################################### 
        X_test_overlay  = []
        for i in range(len(X_test_masks)):
            X_test_overlay.append(X_test_orig[i] * X_test_masks[i])
#########################################################
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
        X_test_swap1 = np.array(X_test_swap1)
        X_test_swap2 = []
        for img in X_test_overlay:
            img = cv2.flip(img,1) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap2.append(s3)
        X_test_swap2   = np.array(X_test_swap2)
        X_test_swap1 = np.array(medfiltering(X_test_swap1))
        X_test_swap2 = np.array(medfiltering(X_test_swap2))
        # X_test_overlay = np.array(X_test_overlay)
        # X_test_swap1   = filtering(X_test_swap1)
        # X_test_swap2   = filtering(X_test_swap2)
        # X_test_f       = filtering(X_test_overlay)  
        res = self.detect_adversarial(X_test_swap1, X_test_swap2, adv_labels_SVM, adv_cls)
        return res
    def detect_adversarial(self,X_test_swap1 ,X_test_swap2,adv_labels,adv_cls):
                adv_final_split1_pred  = self.split_model.predict(X_test_swap1/255.,verbose=0)
                adv_final_split2_pred  = self.split_model.predict(X_test_swap2/255., verbose=0)
                # adv_final_split3_pred  = self.split_model.predict(X_test_f    /255., verbose=0)
                adv_final_split1_Cls   = np.argmax(adv_final_split1_pred, axis=1)
                adv_final_split2_Cls   = np.argmax(adv_final_split2_pred, axis=1)
                # adv_final_split3_Cls   = np.argmax(adv_final_split3_pred, axis=1)
                indeces_eq             = np.where(adv_final_split1_Cls==adv_final_split2_Cls)[0]
                # indeces_eq2            = np.where(adv_final_split1_Cls==adv_final_split3_Cls)[0]
                # l1 = len(indeces_eq1)
                # l2 = len(indeces_eq2)
                # indeces_eq_smal = []
                # indeces_eq_larg = []
                # if l1 < l2:
                #     indeces_eq_smal = indeces_eq1
                #     indeces_eq_larg = indeces_eq2
                # else:
                #     indeces_eq_smal = indeces_eq2
                #     indeces_eq_larg = indeces_eq1
                # indeces_eq = []
                # for i in range(len(indeces_eq_smal)):
                #     if indeces_eq_smal[i] in indeces_eq_larg:
                #         indeces_eq.append(indeces_eq_smal[i])
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
    ##############################################################################################################
    def test_time(self,X_test,Y_test,adv_cls):
        time_all = 0
        Split_start_timeP1 = time.clock()
        X_test_     = []
        X_test_orig = []
        for img in X_test:
            img_uint8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
            image     = cv2.cvtColor(img_uint8,cv2.COLOR_RGB2GRAY)#Image now is 255-level
            image     = np.expand_dims(image, axis=2)
            X_test_orig.append(img_uint8)
            X_test_.append(image)
        X_test_ = np.array(X_test_)
#########################################################
        X_test_U = X_test_/255.
        X_test_mask           = self.U_Net_Model.predict(X_test_U,verbose=0)
        X_test_preds_mask_t   = (X_test_mask > 0.5).astype(np.uint8)
        X_test_masks          = np.array(X_test_preds_mask_t)
        adv_labels_SVM        = np.array(Y_test)
        X_adv_OCsvm           = X_test_masks.reshape((len(X_test_masks), -1))
        result                = self.OC_SVM_model.predict(X_adv_OCsvm )
        oc_indices            = np.where(result==-1)[0]
        adv_labels_SVM[oc_indices] = adv_cls
######################################################### 
        X_test_overlay  = []
        for i in range(len(X_test_masks)):
            X_test_overlay.append(X_test_orig[i] * X_test_masks[i])
#########################################################
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
        X_test_swap1 = np.array(X_test_swap1)
        X_test_swap2 = []
        for img in X_test_overlay:
            img = cv2.flip(img,1) # Mirror
            s3 = np.zeros_like(X_test_overlay[0])
            s3[:, :width_cutoff]=img[:, width_cutoff:]
            s3[:, width_cutoff:]=img[:, :width_cutoff]
            X_test_swap2.append(s3)
        X_test_swap2   = np.array(X_test_swap2)
        X_test_swap1 = np.array(medfiltering(X_test_swap1))
        X_test_swap2 = np.array(medfiltering(X_test_swap2))
        # X_test_overlay = np.array(X_test_overlay)
        # X_test_swap1   = filtering(X_test_swap1)
        # X_test_swap2   = filtering(X_test_swap2)
        # X_test_f       = filtering(X_test_overlay) 
        time_all += (time.clock() - Split_start_timeP1)
        print(time_all, "Split Detection seconds p1 time") 
        res = self.detect_adversarial_time(X_test_swap1, X_test_swap2, adv_labels_SVM, adv_cls, time_all)
        return res
    def detect_adversarial_time(self,X_test_swap1 ,X_test_swap2,adv_labels,adv_cls, time_all):
                Split_start_timeP2 = time.clock()
                adv_final_split1_pred  = self.split_model.predict(X_test_swap1/255.,verbose=0)
                adv_final_split2_pred  = self.split_model.predict(X_test_swap2/255., verbose=0)
                # adv_final_split3_pred  = self.split_model.predict(X_test_f    /255., verbose=0)
                adv_final_split1_Cls   = np.argmax(adv_final_split1_pred, axis=1)
                adv_final_split2_Cls   = np.argmax(adv_final_split2_pred, axis=1)
                # adv_final_split3_Cls   = np.argmax(adv_final_split3_pred, axis=1)
                indeces_eq             = np.where(adv_final_split1_Cls==adv_final_split2_Cls)[0]
                # indeces_eq2            = np.where(adv_final_split1_Cls==adv_final_split3_Cls)[0]
                # l1 = len(indeces_eq1)
                # l2 = len(indeces_eq2)
                # indeces_eq_smal = []
                # indeces_eq_larg = []
                # if l1 < l2:
                #     indeces_eq_smal = indeces_eq1
                #     indeces_eq_larg = indeces_eq2
                # else:
                #     indeces_eq_smal = indeces_eq2
                #     indeces_eq_larg = indeces_eq1
                # indeces_eq = []
                # for i in range(len(indeces_eq_smal)):
                #     if indeces_eq_smal[i] in indeces_eq_larg:
                #         indeces_eq.append(indeces_eq_smal[i])
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
                time_all += (time.clock() - Split_start_timeP2)
                print(time_all, "Split Detection seconds p2 time")
                pred_split_plus   = np.array(pred_split_plus)
                detected_examples   = len(np.where(pred_split_plus==True)[0])
                successful_examples = len(np.where(pred_split_plus==False)[0])
                print("Detection Rate                                         {:.2f}% ".format(detected_examples/pred_split_plus.shape[0]*100))
                print("Successful Rate                                        {:.2f}% ".format(successful_examples/pred_split_plus.shape[0]*100))
                print("Successful Examples                                   ",successful_examples ,"out of", 
                    pred_split_plus.shape[0])
                # return np.where(pred_split_plus==False)[0] , adv_final_seg_Cls, adv_final_inv_Cls,adv_final_seg_pred,adv_final_inv_pred
                return pred_split_plus
    ##############################################################################################################