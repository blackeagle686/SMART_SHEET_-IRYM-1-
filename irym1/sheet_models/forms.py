from django import forms 
from . import models

class PromptForm(forms.Form):
    prompt = forms.CharField(
        required=True,  # or False if you want it optional
        label="What Do you want",
        widget=forms.Textarea(
            attrs={
                "id": "promptInput",
                "class": "",  # add Bootstrap class if needed, e.g., "form-control"
                "rows": 5,
                "placeholder": "What would you like to achieve with your data..."
            }
        )
    )

    
    

