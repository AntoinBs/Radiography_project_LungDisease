import pandas as pd
import numpy as np
import glob
import cv2
import os.path
import Libraries

def LoadMetaD(preload: str= 'yes'):
    ''' This functions loads the metadata of each dataset'''
    if preload=='yes':
        if os.path.isfile(r'..\data\processed\Allmetadata.csv'):
            metadata=pd.read_csv(r'..\data\processed\Allmetadata.csv')
            return metadata
        else: print('The file does not exist')


    else:
        #Load metadata in dataframes
        df_covid = pd.read_excel(r"..\data\raw\COVID\COVID.metadata.xlsx")
        df_lung_opacity = pd.read_excel(r"..\data\raw\Lung_Opacity\Lung_Opacity.metadata.xlsx")
        df_normal = pd.read_excel(r"..\data\raw\Normal\Normal.metadata.xlsx")
        df_viral_pneumonia = pd.read_excel(r"..\data\raw\Viral_Pneumonia\Viral_Pneumonia.metadata.xlsx")
        
        # Add LABELs for datset metadata and concatenate them
        for df, name in zip([df_covid, df_normal, df_viral_pneumonia, df_lung_opacity], ['COVID', 'Normal','Viral_Pneumonia','Lung_Opacity']):
            df['LABEL'] = name
        # Concatenate all metadata together with tag for dataset
        metadata= pd.concat([df_covid, df_normal, df_viral_pneumonia, df_lung_opacity], axis=0, ignore_index=True)
        print(len(metadata))
        
        # Put origins of datasets in dataframe
        origin=[]


        # Create list of simplified origin names
        for url in metadata['URL']:
            if 'rsna' in url:
                origin.append('rsna')
            elif 'bimcv' in url:
                origin.append('bimcv')
            elif 'paultimothymooney' in url:
                origin.append('paultimothy')
            elif 'armiro' in url:
                origin.append('armiro')
            elif 'eurorad' in url:
                origin.append('eurorad')
            elif 'ml_workgroup' in url:
                origin.append('ml_workgroup')
            elif 'ieee8023' in url:
                origin.append('ieee8023')
            else :
                origin.append('sirm')
        
        # Transform list into dataframe
        origin_data=pd.Series(origin)

        # Add simplified origin column to data
        metadata['origin_data']= origin_data
        

        metadata.to_csv(r'..\data\processed\Allmetadata.csv', sep=',', index= False)

    return metadata

def CountWhitePixMasks(preload=str):  

    ''' This function reads masks images from the respective mask folders of all datasets.
    It counts the white pixels in each masks and stores it in a dataframe, the values are then
    divided by the size of the image and the data tagged and put into a signle dataframe.
    
    ------------------------------------------------------
    Arguments
    
    ------------------------------------------------------
    Return
    AllWhitePix dataframe which is n x 3 '''
    if preload=='yes':
        if os.path.isfile(r'..\data\processed\AllwhitePix.csv'):
            Allwhitepix=pd.read_csv(r'..\data\processed\AllwhitePix.csv')
            return Allwhitepix

    else:
        #########################  COVID    
        COVID_white_pix=[]
        for name in glob.glob(r'..\data\raw\COVID\masks\*'):
            COVID_mask=[]
            temp_img= cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            COVID_mask.append(temp_img)
            
            for i in np.arange(len(COVID_mask)):

                COVID_white_pix.append(np.sum(COVID_mask[i] == 255))

            divisor=(COVID_mask[0].shape[0]**2)
            proportion_white_pix_COVID = [x/divisor for x in COVID_white_pix]
        del COVID_mask

        ################### LUNG OPACITY
        LO_white_pix=[]
        for name in glob.glob(r'..\data\raw\Lung_Opacity\masks\*'):
            LO_mask=[]
            temp_img= cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            LO_mask.append(temp_img)
            
            for i in np.arange(len(LO_mask)):

                LO_white_pix.append(np.sum(LO_mask[i] == 255))
            
            divisor=(LO_mask[0].shape[0]**2)
            proportion_white_pix_LO = [x/divisor for x in LO_white_pix]
        del LO_mask

        ######################### NORMAL    
        Normal_white_pix=[]
        for name in glob.glob(r'..\data\raw\Normal\masks\*'):
            Normal_mask=[]
            temp_img= cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            Normal_mask.append(temp_img)
            
            for i in np.arange(len(Normal_mask)):

                Normal_white_pix.append(np.sum(Normal_mask[i] == 255))
            
            divisor=(Normal_mask[0].shape[0]**2)
            proportion_white_pix_Normal = [x/divisor for x in Normal_white_pix]
        del Normal_mask

        ########################### VIRAL PNEUMONIA
        VP_white_pix=[]
        for name in glob.glob(r'..\data\raw\Viral Pneumonia\masks\*'):
            VP_mask=[]
            # Load image
            temp_img= cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            VP_mask.append(temp_img)
            
            for i in np.arange(len(VP_mask)):
                # Count sum of white pixels in image
                VP_white_pix.append(np.sum(VP_mask[i] == 255))

            # Divide sum of white pixels by size of image
            divisor=(VP_mask[0].shape[0]*VP_mask[0].shape[1])
            proportion_white_pix_VP = [x/divisor for x in VP_white_pix]
        del VP_mask

        COVID_whitepix= pd.DataFrame({'count': COVID_white_pix, 'relative': proportion_white_pix_COVID})
        LO_whitepix= pd.DataFrame({'count': LO_white_pix, 'relative': proportion_white_pix_LO})
        Normal_whitepix= pd.DataFrame({'count': Normal_white_pix, 'relative': proportion_white_pix_Normal})
        VP_whitepix= pd.DataFrame({'count': VP_white_pix, 'relative': proportion_white_pix_VP})


        # Change datatypes of columns to float16 to reduce memory usage
        COVID_whitepix.astype({'count': 'int32','relative': 'float16' })
        LO_whitepix.astype({'count': 'int32','relative': 'float16' })
        Normal_whitepix.astype({'count': 'int32','relative': 'float16' })
        VP_whitepix.astype({'count': 'int32','relative': 'float16' })

        # Assign LABEL for each dataframe 
        LABEL= pd.Series('COVID', index=np.arange(len(COVID_whitepix)), name= 'LABEL')
        COVID_whitepix['LABEL']=LABEL

        LABEL= pd.Series('LO', index=np.arange(len(LO_whitepix)), name= 'LABEL')
        LO_whitepix['LABEL']=LABEL

        LABEL= pd.Series('Normal', index=np.arange(len(Normal_whitepix)), name= 'LABEL')
        Normal_whitepix['LABEL']=LABEL

        LABEL= pd.Series('VP', index=np.arange(len(VP_whitepix)), name= 'LABEL')
        VP_whitepix['LABEL']=LABEL

        # Concatenate all datasets with same column names
        Allwhitepix=pd.concat([COVID_whitepix,LO_whitepix,Normal_whitepix,VP_whitepix])


        Allwhitepix.to_csv(r'data\processed\AllwhitePix.csv',sep=',', index= False)

    return Allwhitepix

def LoadImg_flat(metadata, resize: str= 'half', preload: str = 'yes', save: str= None):
    data=[]
    if preload=='yes' and os.path.isfile(r'..\data\processed\df_img_flat.csv')== True:
       
        df=pd.read_csv(r'..\data\processed\df_img_flat.csv')
    elif preload=='yes' and os.path.isfile(r'..\data\processed\df_img_flat_half.csv')== True:
        df=pd.read_csv(r'..\data\processed\df_img_flat_half.csv')
        return df
    else:

        for name in metadata['IMG_URL']:
            temp=cv2.imread(Libraries.get_absolute_path.get_absolute_path(name), cv2.IMREAD_GRAYSCALE)
            
            if resize=='half':
                temp=cv2.resize(temp,(0,0),fx=0.1,fy=0.1)
                    
            print(Libraries.get_absolute_path.get_absolute_path(name))
            temp=temp.reshape(-1)
            data.append(temp)
            del temp

        df=pd.DataFrame(data)
        if save=='yes' and resize== 'half':
            df.to_csv(r'..\data\processed\df_img_flat_half.csv', sep=',', index=False)
        elif save=='yes' and resize== 'no':
            df.to_csv(r'..\data\processed\df_img_flat.csv', sep=',', index=False)
        
    
    return df


def LoadImg(metadata, resize: str= 'half', preload: str = 'yes', save: str= None):
    data=[]
    if preload=='yes' and os.path.isfile(r'..\data\processed\df_img.csv')== True:
       
        df=pd.read_csv(r'..\data\processed\df_img.csv')
        return df
    else:

        for name in metadata['IMG_URL']:
            temp=cv2.imread(Libraries.get_absolute_path.get_absolute_path(name), cv2.IMREAD_GRAYSCALE)
            
            if resize=='half':
                temp=cv2.resize(temp,(0,0),fx=0.1,fy=0.1)
                    
            print(Libraries.get_absolute_path.get_absolute_path(name))
            
            data.append(temp)
            del temp

        
        if save=='yes' and resize== 'half':
            data.to_csv(r'..\data\processed\df_img_flat_half.csv', sep=',', index=False)
        elif save=='yes' and resize== 'no':
            data.to_csv(r'..\data\processed\df_img_flat.csv', sep=',', index=False)
        
    
    return data