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

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            print(input_ids.shape, attention_mask.shape, token_type_ids.shape)
            with torch.no_grad():
                teacher_output = teacher(input_ids, attention_mask, token_type_ids)
            student_output = student(input_ids, attention_mask, token_type_ids)
            
            teacher_start_logits, teacher_end_logits = teacher_output.start_logits, teacher_output.end_logits
            student_start_logits, student_end_logits = student_output.start_logits, student_output.end_logits

            hard_loss_start = hard_loss(student_start_logits, teacher_start_logits.argmax(dim=1))
            hard_loss_end = hard_loss(student_end_logits, teacher_end_logits.argmax(dim=1))

            soft_loss_start = soft_loss(softmax(student_start_logits, dim=1), softmax(teacher_start_logits, dim=1))
            soft_loss_end = soft_loss(softmax(student_end_logits, dim=1), softmax(teacher_end_logits, dim=1))

            loss = ((1 - alpha) * (hard_loss_start + hard_loss_end) + alpha * (soft_loss_start + soft_loss_end)) / 2

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
    print(f"Running on {device} with {num_gpus} GPUs.")
    student = student.model.to(device)
    teacher = teacher.model.to(device)

    if num_gpus > 1:
        student = nn.DataParallel(student, device_ids=range(num_gpus))
        teacher = nn.DataParallel(teacher, device_ids=range(num_gpus))
    
    optimizer = Adam(student.parameters(), lr=5e-5)
    distillation_loop(teacher, student, dataloader, epochs=5, alpha=0.5, device=device)

