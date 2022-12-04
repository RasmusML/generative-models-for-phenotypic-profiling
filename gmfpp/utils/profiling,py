import pandas as pd

# extracting latent variables for each image/cell
def LatentVariableExtraction(metadata, images, batch_size, vae):
    images.shape[0]
    batch_size=batch_size
    batch_offset = np.arange(start=0, stop=images.shape[0]+1, step=batch_size)

    df = pd.DataFrame()
    new_metadata = pd.DataFrame()

    for j, item in enumerate(batch_offset[:-1]):
        start = batch_offset[j]
        end = batch_offset[j+1]

        outputs = vae(images[start:end,:,:,:])
        z = outputs["z"]
        z_df = pd.DataFrame(z.detach().numpy())
        z_df.index = list(range(start,end))
        df = pd.concat([metadata.iloc[start:end], z_df], axis=1)
        new_metadata = pd.concat([new_metadata, df], axis=0)

    return new_metadata

  # Wells Profiles
def well_profiles(nm):
    wa = nm.groupby('Image_Metadata_Well_DAPI').mean().iloc[:,-256:]
    return wa

# function to get the cell closest to each Well profile

def well_center_cells(df,well_profiles,p=2):
    wcc = []
    for w in well_profiles.index:
        diffs = (abs(df[df['Image_Metadata_Well_DAPI'] == w].iloc[:,11:] - well_profiles.loc[w])**p)
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        wcc.append(diffs[diffs_sum == diffs_min].index[0])
    
    return wcc

# Compount/Concentration Profiles
def CC_Profile(nm):
    cc =  nm.groupby(['Image_Metadata_Compound','Image_Metadata_Concentration']).median().iloc[:,-256:]
    return cc

# function to get the cell closest to each Compound/Concentration profile

def cc_center_cells(df,cc_profiles,p=2):
    cc_center_cells = []
    for cc in cc_profiles.index:
        diffs = (abs(df[(df['Image_Metadata_Compound'] == cc[0]) & (nm['Image_Metadata_Concentration'] == cc[1])].iloc[:,-256:] - cc_profiles.loc[cc]))**p
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        cc_center_cells.append(diffs[diffs_sum == diffs_min].index[0])
    
    return cc_center_cells