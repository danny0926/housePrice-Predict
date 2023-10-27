def normalizeCol(df, *names):
    for name in names:
        df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
    return df


def zScoreCol(df, *names):
    for name in names:
        df[name] = df[name].apply(lambda x: (x - x.mean()) / x.std())
    return df


def discreteMap(df, col, epsilon):
    data = df.groupby([col])[['單價']].agg(['mean','median','count'])
    data['index'] = data.index
    sort_data = data.sort_values(data.columns[1],ascending = True)
    
    target = 1
    prevMean = 0.0
    prevMedian = 0.0
    currMean = 0.0
    currMedian = 0.0
    mapping = {}
    
    for index, row in sort_data.iterrows():
        if (prevMean == 0 and prevMedian == 0):
            prevMean = row.loc[('單價', 'mean')]
            prevMedian = row.loc[('單價', 'median')]
        else:
            currMean = row.loc[('單價', 'mean')]
            currMedian = row.loc[('單價', 'median')]
            if ((currMean - prevMean < 0.02) and (currMedian - prevMedian < 0.02)):
                target += 1
            prevMean = currMean
            prevMedian = currMedian
        mapping[index] = target
    return mapping