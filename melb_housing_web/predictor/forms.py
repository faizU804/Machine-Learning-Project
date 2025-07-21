from django import forms

class PredictionForm(forms.Form):
    PROPERTY_TYPES = [
        ('h', 'House'),
        ('u', 'Unit/Apartment'),
        ('t', 'Townhouse'),
    ]
    
    METHOD_CHOICES = [
        ('S', 'Property sold'),
        ('SP', 'Property sold prior'),
        ('PI', 'Property passed in'),
        ('PN', 'Sold prior not disclosed'),
        ('SN', 'Sold not disclosed'),
        ('NB', 'No bid'),
        ('VB', 'Vendor bid'),
        ('W', 'Withdrawn prior to auction'),
        ('SA', 'Sold after auction'),
        ('SS', 'Sold after auction price not disclosed'),
    ]

    REGION_CHOICES = [
        ('Northern Metropolitan', 'Northern Metropolitan'),
        ('Western Metropolitan', 'Western Metropolitan'),
        ('Southern Metropolitan', 'Southern Metropolitan'),
        ('Eastern Metropolitan', 'Eastern Metropolitan'),
        ('South-Eastern Metropolitan', 'South-Eastern Metropolitan'),
        ('Eastern Victoria', 'Eastern Victoria'),
        ('Northern Victoria', 'Northern Victoria'),
        ('Western Victoria', 'Western Victoria'),
    ]
    
    suburb = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Richmond'})
    )
    rooms = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 3'})
    )
    property_type = forms.ChoiceField(
        choices=PROPERTY_TYPES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    method = forms.ChoiceField(
        choices=METHOD_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    seller = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Biggin'})
    )
    date = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    distance = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 10.5', 'step': '0.1'})
    )
    postcode = forms.IntegerField(
        min_value=1000,
        max_value=9999,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 3121'})
    )
    bedrooms = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 3'})
    )
    bathrooms = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 2'})
    )
    cars = forms.IntegerField(
        min_value=0,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 1'})
    )
    landsize = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 290', 'step': '0.1'})
    )
    council_area = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Boroondara'})
    )
    latitude = forms.FloatField(
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., -37.8136', 'step': '0.0001'})
    )
    longitude = forms.FloatField(
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 144.9631', 'step': '0.0001'})
    )
    region = forms.ChoiceField(
        choices=REGION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    property_count = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 4019'})
    ) 