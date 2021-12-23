查看 nohup.out的日志

在 nohup.out 文件目录下输入        tail -fn 50 nohup.out

如何查看 nohup.out 最后几行的日志

在 nohup.out 文件目录下输入        tail -n 50 nohup.out

临时设置GPU

Linux： export CUDA_VISIBLE_DEVICES=1

如果要停止运行，你需要使用以下命令查找到 nohup 运行脚本到 PID，然后使用 kill 命令来删除：

ps -aux | grep "runoob.sh" 

找到 PID 后，就可以使用 kill PID 来删除。

kill -9  进程号PID
