from mmcv.runner import HOOKS, LoggerHook
import swanlab
import torch
import time
from tqdm import tqdm

@HOOKS.register_module()
class SwanLabLoggerHook(LoggerHook):
    def __init__(self, project="mmdet3d", run_name=None, interval=50, 
                 enable_progress_bar=True):
        super().__init__(interval=interval)
        self.project = project
        self.run_name = run_name  # 直接从配置文件读取
        self._swanlab_initialized = False
        self.enable_progress_bar = enable_progress_bar
        self.start_time = None
        self.epoch_start_time = None
        self.total_pbar = None  # 总训练进度条
        self.epoch_pbar = None  # 当前epoch进度条
        self.validation_losses = []
        self.best_val_loss = float('inf')
        self.overfitting_patience = 3  # 连续3个epoch验证损失不下降则认为过拟合
        self.overfitting_counter = 0
        self.total_iters = 0  # 总迭代数
        self.current_iter = 0  # 当前迭代数

    def _init_swanlab(self, runner):
        """初始化SwanLab，使用配置文件中指定的run_name"""
        if self._swanlab_initialized:
            return
        
        # 直接使用配置文件中传入的run_name
        swanlab.init(project=self.project, run_name=self.run_name)
        self._swanlab_initialized = True
        if self.run_name:
            print(f"🚀 SwanLab initialized with run_name: {self.run_name}")

    def before_run(self, runner):
        """训练开始前初始化SwanLab"""
        self._init_swanlab(runner)

    def log(self, runner):
        # 确保SwanLab已初始化
        if not self._swanlab_initialized:
            self._init_swanlab(runner)
        
        # 获取log_buffer中的所有输出
        log_dict = runner.log_buffer.output
        
        # 记录所有数值类型的指标
        metrics = {}
        for k, v in log_dict.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                # 处理单个元素的tensor
                metrics[k] = v.item()
        
        # 检查是否有验证指标
        val_indicators = [k for k in metrics.keys() if 'val' in k.lower()]
        if val_indicators:
            # 处理验证结果
            self._process_validation_results(runner, metrics)
        
        # 如果有指标需要记录，则记录到SwanLab
        if metrics:
            swanlab.log(metrics, step=runner.iter)
        
        # 额外记录一些训练状态信息
        if hasattr(runner, 'epoch'):
            swanlab.log({'epoch': runner.epoch}, step=runner.iter)
        
        # 记录学习率
        if hasattr(runner, 'optimizer') and runner.optimizer is not None:
            for i, param_group in enumerate(runner.optimizer.param_groups):
                if 'lr' in param_group:
                    swanlab.log({f'lr_group_{i}': param_group['lr']}, step=runner.iter)

    def before_train_epoch(self, runner):
        """训练epoch开始前初始化进度条"""
        # 确保SwanLab已初始化
        if not self._swanlab_initialized:
            self._init_swanlab(runner)
        
        if self.enable_progress_bar:
            if self.start_time is None:
                self.start_time = time.time()
                # 初始化总进度条
                self.total_iters = len(runner.data_loader) * runner.max_epochs
                self.total_pbar = tqdm(
                    total=self.total_iters,
                    desc='总训练进度',
                    unit='iter',
                    ncols=120,
                    position=0,
                    leave=True
                )
            
            self.epoch_start_time = time.time()
            
            # 计算当前epoch的迭代数
            epoch_iters = len(runner.data_loader)
            self.epoch_pbar = tqdm(
                total=epoch_iters,
                desc=f'Epoch {runner.epoch + 1}/{runner.max_epochs}',
                unit='iter',
                ncols=120,
                position=1,
                leave=False
            )

    def after_train_epoch(self, runner):
        """训练epoch结束后更新进度条和记录指标"""
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None
        
        # 计算epoch时间
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
        else:
            epoch_time = 0
        
        if self.start_time is not None:
            total_time = time.time() - self.start_time
        else:
            total_time = 0
        
        # 估算剩余时间
        if runner.epoch > 0:
            avg_epoch_time = total_time / (runner.epoch + 1)
            remaining_epochs = runner.max_epochs - (runner.epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
        else:
            estimated_remaining_time = 0
        
        # 记录时间信息到SwanLab
        time_metrics = {
            'epoch_time': epoch_time,
            'total_time': total_time,
            'estimated_remaining_time': estimated_remaining_time
        }
        swanlab.log(time_metrics, step=runner.epoch)
        
        # 简化调试输出
        print(f"\n📋 Epoch {runner.epoch + 1} 训练完成")
        
        # 检查是否有验证结果在log_buffer中
        if hasattr(runner, 'log_buffer') and runner.log_buffer.output:
            log_dict = runner.log_buffer.output
            val_indicators = [k for k in log_dict.keys() if 'val' in k.lower()]
            if val_indicators:
                # 直接处理验证结果
                self._process_validation_results(runner, log_dict)

    def _process_validation_results(self, runner, log_dict):
        """处理验证结果的辅助方法"""
        # 提取验证损失
        val_loss = 0
        val_metrics = {}
        
        # 查找验证相关的指标
        for key, value in log_dict.items():
            if 'val' in key.lower() or 'loss' in key.lower():
                if isinstance(value, (int, float)):
                    val_metrics[key] = value
                    if 'loss' in key.lower() and val_loss == 0:
                        val_loss = value
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    val_metrics[key] = value.item()
                    if 'loss' in key.lower() and val_loss == 0:
                        val_loss = value.item()
        
        # 如果没找到验证损失，尝试其他常见的损失名称
        if val_loss == 0:
            for key, value in log_dict.items():
                if any(loss_name in key.lower() for loss_name in ['loss', 'ce_loss', 'crossentropy', 'focal_loss']):
                    if isinstance(value, (int, float)) and value > 0:
                        val_loss = value
                        val_metrics[f'val_{key}'] = value
                        break
                    elif isinstance(value, torch.Tensor) and value.numel() == 1 and value.item() > 0:
                        val_loss = value.item()
                        val_metrics[f'val_{key}'] = value.item()
                        break
        
        # 如果找到了验证损失，进行过拟合检测
        if val_loss > 0:
            self.validation_losses.append(val_loss)
            
            # 记录验证损失到SwanLab
            val_metrics.update({
                'val_loss': val_loss,
                'val_loss_avg': sum(self.validation_losses) / len(self.validation_losses)
            })
            
            # 检查过拟合
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.overfitting_counter = 0
            else:
                self.overfitting_counter += 1
            
            # 记录过拟合状态
            overfitting_status = {
                'overfitting_counter': self.overfitting_counter,
                'is_overfitting': self.overfitting_counter >= self.overfitting_patience
            }
            val_metrics.update(overfitting_status)
            
            if self.overfitting_counter >= self.overfitting_patience:
                print(f"    ⚠️  警告: 检测到可能的过拟合! 验证损失已连续{self.overfitting_counter}个epoch未下降")
            
            print(f"    ✅ 验证损失: {val_loss:.4f}")
        else:
            # 即使没有找到验证损失，也记录一些基本信息
            val_metrics = {'val_loss': 0, 'validation_error': 1}
        
        # 记录所有验证指标到SwanLab
        if val_metrics:
            print(f"    📝 记录到SwanLab: {val_metrics}")
            swanlab.log(val_metrics, step=runner.epoch)

    def before_val_epoch(self, runner):
        """验证epoch开始前"""
        print(f"\n🚀 开始验证 - Epoch {runner.epoch + 1}")

    def after_val_iter(self, runner):
        """验证迭代后"""
        # 可以在这里记录每个验证batch的损失
        if hasattr(runner, 'outputs') and runner.outputs is not None:
            outputs = runner.outputs
            if isinstance(outputs, dict):
                metrics = {}
                for k, v in outputs.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        metrics[k] = v.item()
                
                if metrics:
                    # 记录验证迭代指标（可选）
                    val_iter_metrics = {f'val_iter_{k}': v for k, v in metrics.items()}
                    swanlab.log(val_iter_metrics, step=runner.iter)

    def after_val_epoch(self, runner):
        """验证epoch结束后记录验证结果"""
        print(f"\n🔍 验证完成 - Epoch {runner.epoch + 1}")
        
        # 从log_buffer获取验证结果
        if hasattr(runner, 'log_buffer') and runner.log_buffer.output:
            log_dict = runner.log_buffer.output
            
            # 提取验证损失
            val_loss = 0
            val_metrics = {}
            
            # 查找验证相关的指标
            for key, value in log_dict.items():
                if 'val' in key.lower() or 'loss' in key.lower():
                    if isinstance(value, (int, float)):
                        val_metrics[key] = value
                        if 'loss' in key.lower() and val_loss == 0:
                            val_loss = value
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        val_metrics[key] = value.item()
                        if 'loss' in key.lower() and val_loss == 0:
                            val_loss = value.item()
            
            # 如果没找到验证损失，尝试其他常见的损失名称
            if val_loss == 0:
                for key, value in log_dict.items():
                    if any(loss_name in key.lower() for loss_name in ['loss', 'ce_loss', 'crossentropy', 'focal_loss']):
                        if isinstance(value, (int, float)) and value > 0:
                            val_loss = value
                            val_metrics[f'val_{key}'] = value
                            break
                        elif isinstance(value, torch.Tensor) and value.numel() == 1 and value.item() > 0:
                            val_loss = value.item()
                            val_metrics[f'val_{key}'] = value.item()
                            break
            
            # 如果找到了验证损失，进行过拟合检测
            if val_loss > 0:
                self.validation_losses.append(val_loss)
                
                # 记录验证损失到SwanLab
                val_metrics.update({
                    'val_loss': val_loss,
                    'val_loss_avg': sum(self.validation_losses) / len(self.validation_losses)
                })
                
                # 检查过拟合
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.overfitting_counter = 0
                else:
                    self.overfitting_counter += 1
                
                # 记录过拟合状态
                overfitting_status = {
                    'overfitting_counter': self.overfitting_counter,
                    'is_overfitting': self.overfitting_counter >= self.overfitting_patience
                }
                val_metrics.update(overfitting_status)
                
                if self.overfitting_counter >= self.overfitting_patience:
                    print(f"⚠️  警告: 检测到可能的过拟合! 验证损失已连续{self.overfitting_counter}个epoch未下降")
                
                print(f"✅ 验证损失: {val_loss:.4f}")
            else:
                # 即使没有找到验证损失，也记录一些基本信息
                val_metrics = {'val_loss': 0, 'validation_status': 'no_loss_found'}
            
            # 记录所有验证指标到SwanLab
            if val_metrics:
                print(f"📝 记录到SwanLab: {val_metrics}")
                swanlab.log(val_metrics, step=runner.epoch)
        else:
            # 记录验证失败状态
            swanlab.log({'val_loss': 0, 'validation_error': 2}, step=runner.epoch)

    def after_train_iter(self, runner):
        """在训练迭代后记录loss值和更新进度条"""
        # 调用父类方法
        super().after_train_iter(runner)
        
        # 更新当前迭代计数
        self.current_iter = runner.iter
        
        # 更新总进度条
        if self.total_pbar:
            self.total_pbar.update(1)
            
            # 计算总进度百分比
            total_progress = (self.current_iter / self.total_iters) * 100
            
            # 计算总预计剩余时间
            if self.current_iter > 0 and self.start_time:
                elapsed = time.time() - self.start_time
                avg_time_per_iter = elapsed / self.current_iter
                remaining_iters = self.total_iters - self.current_iter
                total_eta = avg_time_per_iter * remaining_iters
                
                # 更新总进度条显示
                self.total_pbar.set_postfix({
                    '总进度': f'{total_progress:.1f}%',
                    '总ETA': f'{total_eta/60:.1f}min'
                })
            else:
                # 如果无法计算ETA，只显示进度
                self.total_pbar.set_postfix({
                    '总进度': f'{total_progress:.1f}%'
                })
        
        # 更新epoch进度条
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            
            # 计算并显示epoch预计剩余时间
            if runner.iter > 0 and self.epoch_start_time:
                elapsed = time.time() - self.epoch_start_time
                # 修复除零错误：确保分母不为0
                current_epoch_iter = runner.iter % len(runner.data_loader)
                if current_epoch_iter > 0:
                    avg_time_per_iter = elapsed / current_epoch_iter
                    remaining_iters = len(runner.data_loader) - current_epoch_iter
                    epoch_eta = avg_time_per_iter * remaining_iters
                    self.epoch_pbar.set_postfix({'ETA': f'{epoch_eta:.1f}s'})
                else:
                    self.epoch_pbar.set_postfix({'ETA': '计算中...'})
        
        # 从runner.outputs获取loss值（MMCV的方式）
        if hasattr(runner, 'outputs') and runner.outputs is not None:
            outputs = runner.outputs
            if isinstance(outputs, dict):
                # 处理MMCV的train_step返回的log_vars
                metrics = {}
                for k, v in outputs.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        # 处理单个元素的tensor
                        metrics[k] = v.item()
                    elif isinstance(v, torch.Tensor):
                        # 处理多元素tensor，取平均值
                        metrics[k] = v.mean().item()
                
                # 记录loss值到SwanLab
                if metrics:
                    swanlab.log(metrics, step=runner.iter)

    def after_train(self, runner):
        """训练结束后清理进度条"""
        if self.total_pbar:
            self.total_pbar.close()
            self.total_pbar = None
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None
