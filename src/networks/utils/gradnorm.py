import torch


class GradNorm:
    def __init__(self, model, alpha=1.5):
        self.alpha = alpha
        # 记录每个任务的初始损失
        self.initial_losses = None
        self.model = model
        # 每个任务的权重，初始化为 1
        self.task_weights = torch.nn.Parameter(torch.ones(2, device=device))

    def compute_gradnorm_loss(self, losses):
        """
        losses: list of torch scalar losses [loss_task1, loss_task2]
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in losses], device=device)

        weighted_losses = [self.task_weights[i] * losses[i] for i in range(len(losses))]
        total_loss = sum(weighted_losses)

        # 计算任务的梯度范数
        G_list = []
        for i, l in enumerate(weighted_losses):
            grads = torch.autograd.grad(l, self.model.parameters(), retain_graph=True, create_graph=True)
            G = torch.cat([g.view(-1) for g in grads]).norm()
            G_list.append(G)

        G_avg = sum(G_list) / len(G_list)
        # 计算任务的相对比例
        r_list = [losses[i].item() / self.initial_losses[i].item() for i in range(len(losses))]
        target_G_list = [G_avg * (r ** self.alpha) for r in r_list]

        gradnorm_loss = sum(torch.abs(G_list[i] - target_G_list[i]) for i in range(len(losses)))
        return total_loss, gradnorm_loss
