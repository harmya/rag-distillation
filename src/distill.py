import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm
from dataloader import SQUADataset
from student import Student
from teacher import Teacher

def distillation_loop(teacher=None, student=None, dataloader=None, epochs=5, alpha=0.5, device=None):
    
    if teacher is None or student is None:
        raise ValueError("Teacher and student models must be provided.")
    if dataloader is None:
        raise ValueError("Dataloader must be provided.")
    if device is None:
        # runs on GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    student.train()
    teacher.eval()

    hard_loss = CrossEntropyLoss()
    soft_loss = KLDivLoss(reduction="batchmean")

    for epoch in epochs:
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            with torch.no_grad():
                teacher_output = teacher.model(input_ids, attention_mask, token_type_ids)
            student_output = student.model(input_ids, attention_mask, token_type_ids)
            student_logits = student_output.logits
            teacher_logits = teacher_output.logits

            loss_hard = hard_loss(student_logits, teacher_logits)
            loss_soft = soft_loss(softmax(student_logits, dim=1), softmax(teacher_logits, dim=1))
            loss = alpha * loss_hard + (1 - alpha) * loss_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Loss: {running_loss / len(dataloader)}")
    

if __name__ == "__main__":
    teacher = Teacher()
    student = Student()
    dataset = SQUADataset()
    dataloader = dataset.dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        student = nn.DataParallel(student, device_ids=range(num_gpus))
        teacher = nn.DataParallel(teacher, device_ids=range(num_gpus))
    
    student.to(device)
    teacher.to(device)    
    
    optimizer = Adam(student.parameters(), lr=5e-5)

    distillation_loop(teacher, student, dataloader, epochs=5, alpha=0.5, device=device)

