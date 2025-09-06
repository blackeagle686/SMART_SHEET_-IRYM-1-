from django import forms

class CodePromptForm(forms.Form):
    prompt = forms.CharField(
        required=True,
        label="What do you want",
        widget=forms.Textarea(
            attrs={
                "id": "promptInput",
                "class": "form-control",
                "rows": 5,
                "placeholder": "What would you like to achieve with your data..."
            }
        )
    )

class UserCodeForm(forms.Form):
    user_code = forms.CharField(
        required=False,
        label="Write your Python code",
        widget=forms.Textarea(
            attrs={
                "id": "user_code_input",        # نفس الـ id اللي هنستخدمه في JS
                "class": "form-control code-input",
                "rows": 15,
                "placeholder": "اكتب كود Python هنا...",
            }
        )
    )

class PlotlyChartForm(forms.Form):
    # اختيار نوع الـ Chart
    CHART_KINDS = [
        ("hist", "Histogram"),
        ("box", "Boxplot"),
        ("violin", "Violin"),
        ("kde", "KDE"),
        ("heatmap", "Heatmap"),
        ("pie", "Pie"),
    ]
    kind = forms.ChoiceField(
        choices=CHART_KINDS,
        widget=forms.HiddenInput()
    )

    col = forms.ChoiceField(
        label="Column Name",
        choices=[],  # هيتملأ في init
        widget=forms.Select(attrs={"class": "form-select", "required": True})
    )

    COLOR_MAPS = [
        ("Viridis", "Viridis"),
        ("Cividis", "Cividis"),
        ("Plasma", "Plasma"),
        ("Inferno", "Inferno"),
        ("Magma", "Magma"),
        ("Set3", "Set3 (Categorical)"),
    ]
    colors = forms.ChoiceField(
        choices=COLOR_MAPS,
        label="Color Map",
        widget=forms.Select(attrs={"class": "form-select"})
    )


class ChartSettingsForm(forms.Form):
    chart_count = forms.IntegerField(
        label="Number of Charts",
        min_value=1,
        max_value=10,
        initial=3,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )
    sample_size = forms.IntegerField(
        label="Sample Size",
        min_value=10,
        max_value=10000,
        initial=100,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )