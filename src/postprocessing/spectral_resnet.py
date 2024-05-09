import torch.nn as nn

class SpectralResNetBlocks(nn.Module):

    def __init__(self, n_channels, n_blocks = 1, n_features=32, reduction=4, preprocessing = False):
        super().__init__()
        
        self.spectral_blocks = nn.Sequential(*[SpectralResNet(n_channels, n_features, reduction, preprocessing) for _ in range(n_blocks)])

    def forward(self, x):
        out = self.spectral_blocks(x)
        return out


class SpectralResNet(nn.Module):
    def __init__(self, n_channels, n_features=32, reduction=4, preprocessing = False):
        # TODO: Al parecer en el paper usan de canales de entrada el número de features, pero si la imagen tiene
        # un numero de canales diferente al numero de features peta, asi que simplemente no lo pongo, pero tendré que revisar esto más adelante.

        # Otra cosa a destacar es que reduction ha de ser menor o igual al numero de canales (porque sino la división entera da 0)

        super().__init__()
        self.apply_preprocessing = preprocessing
        self.n_features = n_features if self.apply_preprocessing else n_channels

        self.preprocessing = nn.Conv2d(n_channels, self.n_features, kernel_size=3, padding=1)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.n_features, self.n_features // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features // reduction, self.n_features, 1),
            nn.Sigmoid()
        )

        self.postprocessing = nn.Conv2d(self.n_features, n_channels, kernel_size=3, padding=1)


    def forward(self, x): 
        if self.apply_preprocessing:
            x_features =  self.preprocessing(x)
            output1 = self.feature_extractor(x_features)
            output2 = self.channel_attention(output1)

            product = output2 * output1

            result =  self.postprocessing(product + x_features)
        else:
            output1 = self.feature_extractor(x)
            output2 = self.channel_attention(output1)

            product = output2 * output1

            result =  product + x

        return result