# This is the corrected update_customer function
@app.put('/api/customers/{customer_id}')
async def update_customer(customer_id: int, request: Request):
    """Update a customer"""
    try:
        data = await request.json()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build dynamic update query with all possible fields
        update_fields = []
        values = []
        
        # All possible database fields that can be updated
        all_fields = [
            'CUSTOMER_ID', 'CO_CODE', 'DISTRICTNAME', 'REGIONNAME', 'CBE_REGION', 
            'BRANCHNAME', 'APPROVED_AMOUNT', 'GRANT_DATE', 'EXPIRY_DATE', 'TENURE', 
            'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 'ARRNGEMENT_ID', 
            'RELATIONSHIP_MANAGER', 'LTYPE', 'CUST_SHORTNAME', 'LINE_NO', 
            'ACCT_OFFICER_CODE', 'DAO_NAME', 'DAO_CODE', 'PROD_CODE', 'BUSINESS_DATE', 
            'PRINCIPAL_OS', 'INTEREST_OS', 'PRINCIPAL_ARREARS', 'INTEREST_ARREARS', 
            'CURRENT_COMMITTMENT', 'IS_GOVT_BACKED', 'INTEREST_RATE', 'INSTALLMENT_AMOUNT', 
            'INSTALLMENT_FREQ_PRINCIPAL', 'INSTALLMENT_FREQ_INTEREST', 'RISK_GRADE', 
            'DATE_RATED', 'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 'SECTOR', 
            'TERM_OF_PAYMENT', 'PRODUCT_OWNER', 'LOANID', 'COLLATTERAL', 'COLLATERAL_VALUE', 
            'Latitude', 'Longitude', 'AMOUNT_RANGE', 'COLLATERAL_RANGE', 'fiscal_quarter'
        ]
        
        for field in all_fields:
            if field in data and data[field] is not None:
                update_fields.append(f"{field} = %s")
                values.append(data[field])
        
        if update_fields:
            sql = f"""
            UPDATE customers 
            SET {', '.join(update_fields)}, UPDATED_AT = NOW()
            WHERE id = %s
            """
            values.append(customer_id)
            
            cursor.execute(sql, values)
            conn.commit()
        
        cursor.close()
        conn.close()
        
        return {'success': True, 'message': 'Customer updated successfully'}
        
    except Exception as e:
        logger.error(f"Failed to update customer: {e}")
        return {'success': False, 'error': str(e)}
