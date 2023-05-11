# Deep & Wide Neural Networks

Deep & Wide Model performs both generalisation & memorisation learning both higher & lower order feature interaction. We compared results of three models where Deep-Wide-Network outperformed the other ones.

**I am reviewing the perfomance of Deep & Wide along with other baselines**

![image04](https://github.com/Guggu-Gill/complaint-classification-finance/assets/128667568/9489d45b-97df-4303-9b5d-e9da0293ed5e)


Preprocessing Techniques used-
1. Binning continous variables.
2. One Hot encoding of categorical variable.
3. Encoding text based data using Encoder based attention based Transformer based Uniform Sentence encoder.
![use-transformer-variant](https://github.com/Guggu-Gill/complaint-classification-finance/assets/128667568/a4e8cc18-e1cf-4b9e-a653-94aebca0861e)



Images description
1. Deep-Wide-Network(left),
2. Deep-Network(middle),
3. Wide-Network(left)
![model-diagram (3)](https://user-images.githubusercontent.com/128667568/236314011-e970ebe7-f820-42ad-af0e-6db0dfa083c7.jpg)

Results showing improvement in AUC.  
![AUC](https://user-images.githubusercontent.com/128667568/236313235-7e767740-a91d-4a3e-9d93-3b874ac978ad.jpg)

**clearly 0.59 improvemnt in AUC is significant as compared to baselines. Companies are spending billons of dollars on ML research to improve the AUC by even 0.001.**


