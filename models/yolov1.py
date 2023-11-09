import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, S=7,B=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # 1.
            nn.Conv2d(3, 64, (7,7), 2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 2.
            nn.Conv2d(64, 192, (3,3), 1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 3.
            nn.Conv2d(192, 128, (1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 4.
            nn.Conv2d(128, 256, (3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 5.
            nn.Conv2d(256, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 6.
            nn.Conv2d(256, 512, (3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 7.
            nn.Conv2d(512, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 8.
            nn.Conv2d(256, 512, (3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 9.
            nn.Conv2d(512, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 10.
            nn.Conv2d(256, 512, (3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 11.
            nn.Conv2d(512, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 12.
            nn.Conv2d(256, 512, (3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 13.
            nn.Conv2d(512, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 14.
            nn.Conv2d(256, 512, (3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 15.
            nn.Conv2d(512, 512, (1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 16.
            nn.Conv2d(512, 1024, (3,3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 17.
            nn.Conv2d(1024, 512, (1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 18.
            nn.Conv2d(512, 1024, (3,3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # 19.
            nn.Conv2d(1024, 512, (1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 20.
            nn.Conv2d(512, 1024, (3,3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # # 21.
            # nn.Conv2d(1024, 1024, (3,3), padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),
            # # 22.
            # nn.Conv2d(1024, 1024, (3,3), stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),

            # # 23.
            # nn.Conv2d(1024, 1024, (3,3), padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),
            # # 24.
            # nn.Conv2d(1024, 1024, (3,3), padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),


            # Classifier
            nn.AvgPool2d(7,7)
          )

        # self.head = nn.Sequential(
        #     nn.Linear(50176, 4096),
        #     nn.Linear(4096,S * S * (B * 5 + num_classes)),
        # )
        self.head = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        b,m,h,w = x.shape
        x = self.feature_extractor(x)
        # print(x.shape)
        x = x.view(b, -1)
        x = self.head(x)
        return x

    def __repr__(self):
        repr = {
            "name" : self.__class__.__name__,
            "conf" : "1243",
        }
        return str(repr)



# yolo = YOLOv1(1000)
# im = torch.ones((1,3,224,224))
# o = yolo(im)
# print(o.shape)