import sys
sys.path.append("..")
import os
from utils import parse_config_test
import shutil
from multiprocessing import Process
from test import *
from read_nc import get_name
os.environ["CUDA_VISIBLE_DEVICES"] = "10"  #GPU10 is not exists! So it will use CPU instead!

# we recept there parameters from the shell or user input.
# The first parameter is mode str. The format like this:model=nmc
# The second parameter is time str.The format like this:dateTime=2018072100
# The third parameter is zone str.The format like this:zone=1,2,3,4
# An example:python ReconstructSomeZones.py model=nmc dateTime=2018072100 zone=1,2,3,4. 
# If we don't accept the there parameters.
# we will exit this program.

try:
  mode_str = sys.argv[1].split("=")[1].upper()
  time_str = sys.argv[2].split("=")[1]
  area_str = sys.argv[3].split("=")[1]
except Exception as e:
  print("The parameter is wrong! check it please")
  exit()

l = [int(i) for i in area_str.split(",") if i.isdigit()]
l = list(set(l))
l.sort()

print("-------------------Start to deal with the mode:{0}------------------------".format(mode_str))

# Then We set a loop to processe the accept areas
if len(l) == 0:
  print("There is no area to be processed~~")
  exit()
for i in l:
#for i in range(1, 9):
  i = str(i)
  print("==========================================")
  print("Now we start to process the area {0}".format(i))
 
  # we can get the configuration file name according to the mode str!
  config_name = mode_str + ".conf"

  # The we can read the configuration file according it's path.
  config_file_path = os.path.join(os.getcwd(), "config")
  config_file_path = os.path.join(config_file_path, config_name)
  print("The config file path is:{0}".format(config_file_path))

  # we get the area name according the "i"
  area_name = parse_config_test(config_file_path, i, "area_name")

  # We get the input low resolution(lr) file name and the input dir.
  # Furthermore we can get the low resolution file path.
  input_lr_file = mode_str + "_" + time_str + "_" + area_name + ".nc"
  input_lr_dir = parse_config_test(config_file_path, "input", "input_file_dir_path")
  input_lr_file_path = os.path.join(input_lr_dir, input_lr_file)

  # We can check if the input_lr_file_path exists. We set a flag.
  flag = False
  if os.path.exists(input_lr_file_path):
    print("The low resolution file:{0} is existing!!".format(input_lr_file_path))
    flag = True

   # If the input_lr_file_path is not existing, We continue to process the next area.
  if not flag:
    print("The low resolution file:{0} is not existing!".format(input_lr_file_path))
    continue

  # Here we can gurantee the lr file exists. We can start our deepsd.
  # Firstly we copy the lr file to the tmp_testset dir. 
  # An example:/home/pwq/models_simplify/tmp_testset/NMC_testset/2018062400/1
  test_set_name = mode_str + "_" + "testset" 
  test_set_path = os.path.join(os.getcwd(), "tmp_testset")
  test_set_path = os.path.join(test_set_path, test_set_name)

  test_set_date_path = os.path.join(test_set_path, time_str)
  test_set_path = os.path.join(test_set_date_path, i)
  print("The test set path of this area is:{0}".format(test_set_path))

  tmp_output_name = mode_str + "_" + "tmp_output" 
  tmp_output = os.path.join(os.getcwd(), "tmp_output")
  tmp_output = os.path.join(tmp_output, tmp_output_name)
  tmp_output_date_path = os.path.join(tmp_output, time_str)
  tmp_output = os.path.join(tmp_output_date_path, i)
  print("The tmp output path of this area is:{0}".format(tmp_output))
 
  # And we re-create the test_set_path to accept the lr file.
  if not os.path.exists(test_set_path):
    os.makedirs(test_set_path) 
  shutil.copy(input_lr_file_path, test_set_path)

  # If you want to delete the original lr file, Open the next line's annotation.
  #os.remove(input_lr_file_path)

  # we get the final file name via the file in the test_set_path
  try:
    final_file_name = get_name(test_set_path)
  except Exception as e:
    print("During get the final file name, There is a error:", e)

  # We get the output dir according to the config to save the final hr file.
  output_hr_dir = parse_config_test(config_file_path, "output", "output_file_dir_path")
  yy = time_str[:4] 
  yymmdd = time_str[:8]

  output_hr_dir_path = os.path.join(output_hr_dir, yy)
  output_hr_dir_path = os.path.join(output_hr_dir_path, yymmdd)

  # we get the final hr file path via the output_hr_dir_path and the names
  final_hr_abspath = os.path.join(output_hr_dir_path, "".join(final_file_name))
  print("The final hr file aspath is:{0}".format(final_hr_abspath))
  
  # If the final hr file is exists. we continue the next loop.
  if os.path.exists(final_hr_abspath):
    print("The final sp1 file:{0} is existing!".format(final_file_name))
    shutil.rmtree(test_set_date_path)
    continue
  # If the final output dir is not existing, We create it.
  if not os.path.exists(output_hr_dir_path):
      os.makedirs(output_hr_dir_path)

  if flag:
    print("the lr file to be processed is:{0}".format(input_lr_file_path))
    print("the output hr file of area is stored in:{0}".format(output_hr_dir_path))

    try:
      p=Process(target=run_test, args=(config_file_path,test_set_path, i, mode_str, tmp_output))
      p.start() 
    except Exception as e:
      print("There is a error:", e)
    p.join()

    print("The area{0}'s sp1 finished!".format(i))
    print("Now we sent the tmp output file to the final output dir.......")

    # The final output file now is in the temp output dir.For example, In the mode EC, the
    # final output file is in the output3 because there are three layers in the EC model.
    # Which means the final output file is in the locdir[-1].

    f = os.listdir(tmp_output)
    print("The file in temp_output is:{0}".format(f))
    if len(f)>1:
      print("The temp output dir:{0} has more than one file! please check it!".format(tmp_output))
      exit()
    elif len(f) == 0:
      print("The temp output dir:{0} has no file! please check it!!".format(tmp_output))
      exit()
    else:
      files = f[0]

    send_file_path = os.path.join(tmp_output, files)
    print("The send file path is:{0}".format(send_file_path))
    shutil.copy(send_file_path, output_hr_dir_path)
    print("Send the file to final dir complete!")

    print("start to delete the tmp dir in the project......")
    print("deleting the test set dir:{0}".format(test_set_date_path))
    print("deleting the tmp output dir:{0}".format(tmp_output_date_path))
    try:
      shutil.rmtree(test_set_date_path)
      shutil.rmtree(tmp_output_date_path)
    except Exception as e:
      print("During deleting the tmp dir, There is a error:",e)
  else:
    print("No new update files like {0}".format(time_str))
    exit()

