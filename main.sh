
first=0
while :; do

  #gnome-terminal -- sh -c "python3 /home/eii/scala/scara_robot/src/main_gui.py ; exit; bash"  &
  #echo "yo"
     if [ $first == 0 ]; then

	if pgrep -x "python3"  > /dev/null ;then
	    flag=false
	else
	    gnome-terminal -- sh -c "python3 /home/eii/scala/scara_robot/src/main_gui.py ; exit; bash"
        first=1
        sleep 3s
	pkill -f main_gui.py
	fi
     else
#sleep 10s

	if pgrep -x "python3" > /dev/null ;then
	    flag=false
	else
	    gnome-terminal -- sh -c "python3 /home/eii/scala/scara_robot/src/main_gui.py ; exit; bash"
	fi
    


    fi

#sleep 2
done
