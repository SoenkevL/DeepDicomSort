# Dataset information

## General
I use a function which based on the name of the name of a scan assigns it a label from a labelmap

Labelmap:


general decision function:
```azure
def assignLabel(dataframe):
    description = dataframe['description']
    try:
        weight = dataframe['weighting']
    except:
        weight = None
    otherCategories = ['fieldmap','scout','calibration','phase','localizer','nan','NotMri','pet','CT','e2']
    if description in otherCategories:
        return 17
    if description =='T1w':
        return 0
    if description =='T1c':
        return 1
    if weight:
        if description =='T2w':
            if weight=='T2':
                return 2
            if weight=='PD':
                return 5
    if description == 'T2w':
        return 2
    if 'FLAIR' in description:
        return 3
    if description =='FS':
        return 4
    if description =='pd':
        return 5
    if description =='swi' or description=='minIP':
        return 6
    if description.lower() =='gre':
        return 7
    if description =='T2star':
        return 8
    if description =='dwi':
        return 9
    if description =='adc':
        return 10
    if description =='bold':
        return 11
    if description =='angio': #flowsensitive
        return 12
    if description =='pwi' or description=='cbf':
        return 13
    if 'asl' in description:
        return 14
    if description == 'hippo':
        return 15
    if description == 'dti':
        return 16
    return -1
```
this function is used after individual name unification inside of the datasets to assign labels

### ADNI
name unification function for the adni datasets:
```azure
def renameDesc(desc):
    desc = desc.lower()
    if 'localizer' in desc:
        return 'localizer'
    if 'phase' in desc:
        return 'phase'
    if 'field' in desc:
        return 'fieldmap'
    if 'rage' in desc or 'fspgr' in desc:
        return 'T1w'
    if 'cal' in desc:
        return 'calibration'
    if 'scout' in desc:
        return 'scout'
    if 'flair' in desc:
        return 'T2-FLAIR'
    if 'star' in desc:
        return 'T2star'
    if 'swi' in desc:
        return 'swi'
    if 'hippo' in desc:
        return 'hippo'
    if 'perf' in desc or 'pwi' in desc:
        return 'pwi'
    if 'asl' in desc:
        return 'asl'
    if 'dti' in desc:
        return 'dti'
    if 'cerebral_blood_flow' in desc:
        return 'cbf'
    if 't2' in desc:
        return 'T2w'
```

### oasis

In oasis I can read the labels directly from the last part of the name

### egd

```azure
def rename(origName):
    on = origName.lower()
    if 'local' in on:
        return 'localizer'
    if 'gd' in on:
        if 't1' in on or 'fspgr' in on:
            return 'T1c'
    if 't1' in on or 'fspgr' in on:
        return 'T1w'
    if 'flair' in on:
        return 'flair'
    if 'dwi' in on:
        return 'dwi'
    if 't2' in on:
        return 'T2w'
    if 'adc' in on:
        return 'adc'
    if 'pwi' in on or 'perf' in on:
        return 'pwi'
    if 'pd' in on:
        return 'pd'
    if 'dti' in on:
        return 'dti'
    if 'survey' in on:
        return 'survey'
    if 'asl' in on:
        return 'asl'
    return 'other'
```