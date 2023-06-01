import wesad_pretraining
import wesad_finetuning
import wesad_supervised
import statistics as stats

def ssl(subject=0):
	print("WESAD PRETRAINING")
	pretrain_checkpoints_dir = wesad_pretraining.init_wandb(name=f"PT_{subject}")
	wesad_pretraining.train(subject)
	arr = [0.95, 0.98, 0.99, 0.9925, 0.995]
	#arr = [0.95]
	finetuning = []
	supervised = []
	for remove_percent in arr:
		acc_arr, acc1_arr, f1_arr, f11_arr = [], [], [], []
		sv_acc_arr, sv_acc1_arr, sv_f1_arr, sv_f11_arr = [], [], [], []
		for j in range(5):
			print("remove percent = {}, finetuning, step {}".format(remove_percent, j))
			wesad_finetuning.init_wandb(name=f"FT_{subject}_{remove_percent}_{j}")
			tr_data = wesad_finetuning.train(subject, remove_percent=remove_percent, pretrain_checkpoints_dir=pretrain_checkpoints_dir)
			acc, acc1, f1, f11 = wesad_finetuning.test(subject, remove_percent=remove_percent)
			acc_arr.append(acc)
			acc1_arr.append(acc1)
			f1_arr.append(f1)
			f11_arr.append(f11)
			print("remove percent = {}, supervised, step {}".format(remove_percent, j))
			wesad_supervised.init_wandb(name=f"SV_{subject}_{remove_percent}_{j}")
			wesad_supervised.train(subject, remove_percent=remove_percent, train_dataset=tr_data)
			acc, acc1, f1, f11 = wesad_supervised.test(subject, remove_percent=remove_percent)
			sv_acc_arr.append(acc)
			sv_acc1_arr.append(acc1)
			sv_f1_arr.append(f1)
			sv_f11_arr.append(f11)
			#print(acc, acc1, f1, f11)
		finetuning.append([(stats.mean(acc_arr), stats.stdev(acc_arr)), (stats.mean(acc1_arr), stats.stdev(acc1_arr)), (stats.mean(f1_arr), stats.stdev(f1_arr)), (stats.mean(f11_arr), stats.stdev(f11_arr))])
		supervised.append([(stats.mean(sv_acc_arr), stats.stdev(sv_acc_arr)), (stats.mean(sv_acc1_arr), stats.stdev(sv_acc1_arr)), (stats.mean(sv_f1_arr), stats.stdev(sv_f1_arr)), (stats.mean(sv_f11_arr), stats.stdev(sv_f11_arr))])
	print("FINETUNING")
	for i in range(len(finetuning)):
		print(arr[i], ":", finetuning[i])
	print("SUPERVISED")
	for i in range(len(supervised)):
		print(arr[i], ":", supervised[i])

ssl()
