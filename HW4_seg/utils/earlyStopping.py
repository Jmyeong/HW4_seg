import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="checkpoint.pth"):
        """
        EarlyStopping 초기화
        :param patience: 성능 향상이 없을 때 기다리는 에폭 수
        :param delta: 성능 향상이 있어야 하는 최소 변화량
        :param save_path: 모델을 저장할 경로
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_model_wts = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, train_loss, val_loss, model):
        """
        EarlyStopping이 진행되는 함수
        :param train_loss: 학습 데이터에 대한 손실
        :param val_loss: 검증 데이터에 대한 손실
        :param model: 현재 학습된 모델
        """
        # 검증 로스가 개선되지 않거나, 학습 로스가 검증 로스보다 과도하게 감소하는 경우
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.best_model_wts = model.state_dict()  # 모델의 가중치 저장
            self.counter = 0  # 성능 향상이 있으면 카운터 리셋
        elif train_loss < self.best_train_loss - self.delta:
            self.best_train_loss = train_loss
            self.counter += 1
            if self.counter >= self.patience:  # patience만큼 성능 향상이 없다면 early stop
                self.early_stop = True

    def load_best_model(self, model):
        """최고의 모델 가중치를 불러오는 함수"""
        model.load_state_dict(self.best_model_wts)


