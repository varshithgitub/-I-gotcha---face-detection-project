#1 

#this is the change
def val(ip_list):
    return [ i for i in ip_list if i%2==0]

ip = input("enter the no. in a sequence")

ip_list = list(map(int, ip.split()))
print("even numbers are",val(ip_lis))
