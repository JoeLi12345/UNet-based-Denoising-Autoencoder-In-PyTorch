import wesad_pretraining
import wesad_finetuning

def ssl(subject=2):
	print("WESAD PRETRAINING BEGIN")
	pretrain_checkpoints_dir = wesad_pretraining.init_wandb()
	wesad_pretraining.train(subject)
	'''print("WESAD FINETUNING BEGIN")
	wesad_finetuning.init_wandb()
	wesad_finetuning.train(subject, pretrain_checkpoints_dir=pretrain_checkpoints_dir)
	acc, acc1, f1, f11 = wesad_finetuning.test(subject)
	return acc, acc1, f1, f11'''

'''for subject in range(2, 3):
	print(ssl(subject)'''
