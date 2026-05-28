async function addCustomer() {
            const formData = new FormData(document.getElementById('addCustomerForm'));
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/customers', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Customer added successfully!');
                    closeAddCustomerModal();
                    loadCustomers(); // Refresh list
                } else {
                    alert('Error adding customer: ' + result.error);
                }
            } catch (error) {
                console.error('Error adding customer:', error);
                alert('Failed to add customer');
            }
        }
