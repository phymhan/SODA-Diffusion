import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os, subprocess, time

gpus = [str(i) for i in range(8)]

def send_email():
    # Sender and receiver email addresses
    from_email = "cqueenccc@gmail.com"
    to_email = "cqueenccc@gmail.com"
    password = "tihc mixz ioaw smfc"
    
    # Email content
    subject = "Test Email"
    body = "This is a test email sent from Python script."

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # Create server object with SSL option
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

    # Perform operations via server
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

    print("Email sent successfully")

if __name__ == "__main__":
    def check_gpu_usage():
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        # Split the output by lines
        lines = output.split('\n')
        
        # Look for the start of the processes table
        processes_start = False
        for line in lines:
            if '| Processes:' in line:
                processes_start = True
                continue
            
            if processes_start:
                # split line by one or multiple spaces
                fields = line.split()
                if len(fields) < 3:
                    continue
                gpu_id = fields[1]
                # pop this gpu_id from gpus list
                if gpu_id in gpus:
                    gpus.remove(gpu_id)
        
        if len(gpus) == 0:
            return False
        else:
            return True

    # Main monitoring loop
    while True:
        if check_gpu_usage():
            send_email()
            time.sleep(120) 
            if check_gpu_usage():
                send_email()
            exit(0)
        else:
            print("GPU is in use")
            time.sleep(60)
        
