"""fix token data schema

Revision ID: 8727fdd646e2
Revises: add_token_history_tables
Create Date: 2024-03-21

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8727fdd646e2'
down_revision = 'add_token_history_tables'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop existing columns that might cause conflicts
    try:
        op.drop_column('token_data', 'price_change_24h')
    except:
        pass
    try:
        op.drop_column('token_data', 'high_24h')
    except:
        pass
    try:
        op.drop_column('token_data', 'low_24h')
    except:
        pass
    
    # Add columns back with correct schema
    op.add_column('token_data', sa.Column('price_change_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('high_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('low_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    # Drop columns
    op.drop_column('token_data', 'price_change_24h')
    op.drop_column('token_data', 'high_24h')
    op.drop_column('token_data', 'low_24h')
    op.drop_column('token_data', 'last_updated') 