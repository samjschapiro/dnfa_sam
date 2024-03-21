import torch
import numpy as np
import projgrad
import cupy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ssam_obj_func(beta, nabla_f, nabla_l, lam):
    # nabla_f = args[0]
    # nabla_l = args[1] 
    # lam = args[2]
    beta, nabla_f, nabla_l = np.ravel(beta), np.ravel(nabla_f), np.ravel(nabla_l)
    f = - np.inner(beta, nabla_l) - lam*(np.inner(beta,nabla_f))**2

    grad = -nabla_l - lam*2 * nabla_f * np.inner(nabla_f, beta)
    return f, grad
    

class SSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, lam=1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.nabla_f = {}
        self.lam = lam
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, n_iter=25, lam=1):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                nabla_f = self.nabla_f[p]
                nabla_l = p.grad
                res = projgrad.minimize(ssam_obj_func, x0=np.ravel(cupy.array(p.grad.cpu().numpy())), 
                                        rho=self.rho, args=(cupy.array(nabla_f.cpu().numpy()), cupy.array(nabla_l.cpu().numpy()), self.lam), 
                                        maxiters=n_iter, algo=None)
                e_w = res.x
                p.add_(torch.Tensor(np.reshape(e_w, p.shape)).to(device))  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def prep(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.nabla_f[p] = p.grad
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


    def projected_gradient_descent(params, loss_fn, epsilon, num_steps, step_size, projection_fn, *args, **kwargs):
        """
        Projected Gradient Descent optimization algorithm.

        Args:
            params (torch.Tensor): Parameters to optimize.
            loss_fn (callable): Function to compute the loss.
            epsilon (float): Maximum allowed perturbation.
            num_steps (int): Number of optimization steps.
            step_size (float): Step size for the optimization update.
            projection_fn (callable): Function to project the parameters onto a feasible set.
            *args, **kwargs: Additional arguments to pass to loss_fn.

        Returns:
            torch.Tensor: Optimized parameters.
        """
        for _ in range(num_steps):
            params.requires_grad = True
            loss = loss_fn(params, *args, **kwargs)
            grad = torch.autograd.grad(loss, params)[0]
            params = params - step_size * grad
            params = projection_fn(params, epsilon)
        return params

    def projection_l2_ball(params, epsilon):
        """
        Project parameters onto the L2 ball.

        Args:
            params (torch.Tensor): Parameters to project.
            epsilon (float): Radius of the L2 ball.

        Returns:
            torch.Tensor: Projected parameters.
        """
        norm = torch.norm(params)
        if norm > epsilon:
            params = params * epsilon / norm
        return params
    

    def second_sam_inner_update(params, *args, **kwargs):
        """
        Compute second sam inner update rule

        Args:
            params
            func_grads
            

        Returns:
            np.float64: Loss
        """